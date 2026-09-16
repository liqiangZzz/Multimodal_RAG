"""
多模态嵌入处理模块（DashScope 云端版）

功能概述
--------
封装阿里云百炼（DashScope）的 multimodal-embedding-v1 模型，
为文本、图片、以及「文本+图片」三类输入生成稠密向量（dense embedding）。
生成的向量维度由 DashScope 决定（1024 或 768），可用于后续的
向量检索、语义搜索、RAG 等场景。

支持三种输入模式
----------------
1. 'text'        : 纯文本
2. 'image'       : 纯图片（本地文件或 URL）
3. 'text_image'  : 文本 + 图片（同一次请求，向量融合两种模态）

主要能力
--------
- 图片输入规范化：
    - URL      : 发送 HEAD 请求检查可达性与体积，超限或不可达则跳过
    - 本地文件 : 转为 base64 data URI
- API 调用封装：统一处理成功/失败响应、解析 embedding、提取 Retry-After
- RPM 限流：固定窗口限流器，避免超过 DashScope 每分钟调用上限
- 429 重试：调用方可根据返回的 _status / _retry_after 做指数退避重试
- 字段透传：在返回的 dict 中附带 _status 和 _retry_after，便于上层判断

对外主要接口
------------
- image_to_base64(img_path)          : 本地图片 -> base64 data URI
- normalize_image(img)               : 图片输入规范化（URL / 本地文件）
- call_dashscope_once(input_data)    : 单次调用 DashScope API
- process_item_with_guard(item, ...) : 处理单个数据项，生成嵌入向量
- build_work_items(expanded_data)    : 把原始数据拆分为待处理的工作项

返回字段约定
------------
process_item_with_guard 返回的 dict 中会包含：
- dense        : List[float]   嵌入向量（失败时为空列表）
- _status      : Optional[int] HTTP 状态码（成功 200，失败可能为 429/401/None）
- _retry_after : Optional[float] 服务端建议的重试等待秒数（若有）

注意
----
- 该模块只负责“向量化”，不负责写库、检索、分块等操作。
- 使用前需在环境中配置 DASHSCOPE_API_KEY（见 utils/env_utils.py）。
- 若需要本地部署模型，请改用 gme_qwen2_vl_2b_embedding 模块。
"""

import os
import time
from http import HTTPStatus
from typing import Tuple, List, Dict, Optional

import dashscope

from utils.env_utils import ALIBABA_API_KEY
from utils.log_utils import log

# ========= 配置区 =========
DASHSCOPE_MODEL = "multimodal-embedding-v1"  # 指定使用的达摩院多模态嵌入模型名称

# 每分钟最多调用次数（Requests Per Minute）
RPM_LIMIT = 120
# 限流时间窗口（秒），与RPM_LIMIT配合实现每分钟限流
WINDOW_SECONDS = 60

# 是否在遇到429（请求过多）状态码时进行重试
RETRY_ON_429 = True
# 429状态码的最大重试次数
MAX_429_RETRIES = 5
# 指数退避算法的基础等待时间（秒）
BASE_BACKOFF = 2.0

# 图片最大体积（URL HEAD 检查），若超过则跳过图片项
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB

# 是否把同一 item 里的「文本 + 图片」合并为一次 API 调用
# True  -> 走 'text_image' 模式，一次请求得到融合向量
# False -> 拆成 'text' 和 'image' 两次请求，分别得到独立向量
COMBINE_TEXT_IMAGE = True


# ======== 配置区结束 =========


class FixedWindowRateLimiter:
    """固定窗口速率限制器类，用于控制API调用频率"""

    def __init__(self, limit: int, window_seconds: int):
        """初始化速率限制器

        Args:
            limit: 时间窗口内允许的最大请求数
            window_seconds: 时间窗口长度（秒）
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_start = time.monotonic()  # 当前时间窗口的开始时间
        self.count = 0  # 当前时间窗口内的请求计数

    def acquire(self):
        """获取一次请求许可。

        如果当前窗口内请求数已达上限，会阻塞直到下一个窗口开始。
        """
        now = time.monotonic()
        elapsed = now - self.window_start  # 计算当前时间窗口已过去的时间

        # 窗口已过期：重置计数器和窗口开始时间
        if elapsed >= self.window_seconds:
            self.window_start = now
            self.count = 0

        # 如果当前窗口内请求数已达到限制，等待到下一个窗口
        if self.count >= self.limit:
            sleep_sec = self.window_seconds - elapsed  # 计算需要等待的时间
            if sleep_sec > 0:
                print(f"[限速] 达到 {self.limit} 次请求，等待 {sleep_sec:.2f}s...")
                time.sleep(sleep_sec)  # 阻塞等待
            # 等待后重置计数器和窗口开始时间
            self.window_start = time.monotonic()
            self.count = 0

        self.count += 1  # 增加请求计数


# 全局速率限制器实例（所有 API 调用共用）
limiter = FixedWindowRateLimiter(RPM_LIMIT, WINDOW_SECONDS)


def image_to_base64(img: str) -> Tuple[str, str]:
    """将本地图片文件转换为 base64 data URI。

    Args:
        img: 本地图片路径

    Returns:
        (api_image, store_image)
        - api_image: 形如 "data:image/png;base64,xxxx" 的字符串，供 API 使用
        - store_image: 原始图片路径，供后续入库/展示使用
        失败时返回 ("", "")
    """

    try:
        import base64, mimetypes
        # 猜测文件MIME类型，默认 image/png
        mime = mimetypes.guess_type(img)[0] or "image/png"
        # 读取文件并编码为base64
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 构建data URI格式
        return f"data:{mime};base64,{b64}", img
    except Exception as e:
        print(f"[图片] 本地文件转 base64 失败：{e}")
        log.exception(e)
        return "", ""


def normalize_image(img: str) -> Tuple[str, str]:
    """规范化图像输入，统一处理 URL 和本地文件。

    处理逻辑：
        - URL      : 发送 HEAD 请求检查可达性与体积，超限或不可达则跳过
        - 本地文件 : 转为 base64 data URI
        - 其他     : 返回空

    Args:
        img: 图像路径或 URL 字符串

    Returns:
        (api_image, store_image)
        - api_image: 用于 API 向量化的图像数据
        - store_image: 用于存储/展示的图像标识（URL 原值或本地路径）
        无效或超限时返回 ("", "")
    """

    if not img:
        return "", ""

    raw = img.strip()  # 去除首尾空格
    low = raw.lower()  # 转换为小写便于判断

    # URL处理
    if low.startswith("http://") or low.startswith("https://"):
        try:
            import requests
            # 发送HEAD请求获取图像信息（不下载正文）
            head = requests.head(raw, timeout=5, allow_redirects=True)
            if head.status_code != 200:
                print(f"[图片] URL 不可达，status {head.status_code}：{raw}")
                return "", ""

            # 检查 Content-Length，若超过阈值则跳过
            size = int(head.headers.get("Content-Length") or 0)
            if size and size > MAX_IMAGE_BYTES:
                print(f"[图片] URL 大小 {size} > {MAX_IMAGE_BYTES}，跳过该图：{raw}")
                return "", ""
        except Exception as e:
            print(f"[图片] HEAD 检查异常：{e}")
        # API 用 URL；store 用 URL 原值
        return raw, raw

    # 本地文件处理
    if os.path.isfile(raw):
        return image_to_base64(raw)

    # 其他不支持的类型
    return "", ""


def call_dashscope_once(input_data: List[Dict]) -> Tuple[bool, List[float], Optional[int], Optional[float]]:
    """调用 DashScope 多模态嵌入 API 一次。

    会自动应用全局限流；调用失败时返回空向量。

    Args:
        input_data: API 输入列表，元素为 dict，例如：
                    [{"text": "..."}]
                    [{"image": "data:image/png;base64,..."}]
                    [{"text": "..."}, {"image": "..."}]

    Returns:
        (success, embedding, status_code, retry_after)
        - success: 是否成功
        - embedding: 嵌入向量（失败时为空列表）
        - status_code: HTTP 状态码（异常时为 None）
        - retry_after: 服务端建议的重试等待秒数（若有）
    """

    # 应用速率限制
    limiter.acquire()

    try:
        # 调用达摩院多模态嵌入API
        response = dashscope.MultiModalEmbedding.call(
            model=DASHSCOPE_MODEL,
            input=input_data,
            api_key=ALIBABA_API_KEY
        )
    except Exception as e:
        print(f"调用 DashScope 异常：{e}")
        log.exception(e)
        return False, [], None, None

    # 获取HTTP状态码
    status = getattr(response, "status_code", None)
    retry_after: Optional[float] = None

    # 尝试从响应头读取 Retry-After 字段
    try:
        headers = getattr(response, "headers", None)
        if headers and isinstance(headers, dict):
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                retry_after = float(ra)
    except Exception as e:
        pass
        # log.exception(e)

    # 获取API返回的代码和消息
    resp_code = getattr(response, "code", "")
    resp_msg = getattr(response, "message", "")

    # 处理成功响应
    if status == HTTPStatus.OK:
        try:
            # 提取嵌入向量，响应结构：output.embeddings[0].embedding
            embedding = response.output['embeddings'][0]['embedding']
            return True, embedding, status, retry_after
        except Exception as e:
            print(f"解析嵌入失败：{e}")
            log.exception(e)
            return False, [], status, retry_after
    else:
        # 处理失败响应
        print(f"请求失败，状态码：{status}，code：{resp_code}，message：{resp_msg}")
        return False, [], status, retry_after


def process_item_with_guard(
        item: Dict,
        mode: str,
        api_image: str = "",
) -> Dict:
    """处理单个数据项，生成嵌入向量。

    支持模式：
        - 'text'       : 纯文本
        - 'image'      : 纯图片
        - 'text_image' : 文本 + 图片（同一次请求，向量融合）

    Args:
        item: 原始数据项
        mode: 处理模式
        api_image: 当 mode 为 'image' 或 'text_image' 时使用的图像数据

    Returns:
        处理后的数据项副本，新增字段：
            - 'dense': 嵌入向量（失败时为空列表）
    """
    new_item = item.copy()
    raw_text = (new_item.get("text") or "").strip()

    # ---- 统一构造 input_data，避免多处 return 漏设字段 ----
    input_data: Optional[List[Dict]] = None

    if mode == "text":
        if raw_text:
            input_data = [{"text": raw_text}]

    elif mode == "image":
        if api_image:
            input_data = [{"image": api_image}]

    elif mode == "text_image":
        if raw_text and api_image:
            input_data = [{"text": raw_text, "image": api_image}]
        elif raw_text:
            input_data = [{"text": raw_text}]
        elif api_image:
            input_data = [{"image": api_image}]

    # ---- 无有效输入：直接返回空结果，但仍然带上状态字段 ----
    if not input_data:
        new_item["dense"] = []
        new_item["_status"] = None
        new_item["_retry_after"] = None
        return new_item

    # ---- 云端推理 ----
    ok, embedding, status, retry_after = call_dashscope_once(input_data)

    new_item["dense"] = embedding if ok else []
    new_item["_status"] = status
    new_item["_retry_after"] = retry_after

    # 纯图片模式且无文本时，补占位文本
    if mode == "image" and not raw_text:
        new_item["text"] = "图片"

    return new_item


def build_work_items(expanded_data: List[Dict]) -> List[Tuple[Dict, str, str]]:
    """构建工作项列表。

    每个工作项为三元组 (item, mode, api_image)：
        - item      : 待处理的数据项（已复制，避免污染原数据）
        - mode      : 'text' / 'image' / 'text_image'
        - api_image : 已规范化的图像数据（纯文本模式为空字符串）

    合并逻辑由 COMBINE_TEXT_IMAGE 控制：
        - True 且同时有 text 和 image -> 合并为一次 'text_image' 调用
        - 否则拆分为 'text' 和 'image' 两次调用
    """
    work_items: List[Tuple[Dict, str, str]] = []

    for item in expanded_data:
        content = (item.get('text') or '').strip()  # 获取文本内容
        image_raw = (item.get('image_path') or '').strip()  # 获取原始图像路径

        # 规范化图片（可能返回空，表示无效或超限）
        api_img, store_img = ("", "")
        if image_raw:
            api_img, store_img = normalize_image(image_raw)

        # 合并模式：文本 + 图片 一次搞定
        if COMBINE_TEXT_IMAGE and content and api_img:
            combo_item = item.copy()
            combo_item["image_path"] = store_img
            work_items.append((combo_item, "text_image", api_img))
            continue

        # 非合并模式（或只有单边）
        if content:
            work_items.append((item, "text", ""))

        if api_img:
            pic_item = item.copy()
            pic_item["image_path"] = store_img
            work_items.append((pic_item, "image", api_img))

    return work_items


if __name__ == "__main__":
    pass
