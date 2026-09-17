"""
多模态嵌入处理模块（本地部署版）

使用本地部署的 Alibaba-NLP/gme-Qwen2-VL-2B-Instruct 模型，
替代原有的 DashScope 云端 multimodal-embedding-v1 API。

支持三种输入模式：
    - 'text'       : 纯文本
    - 'image'      : 纯图片
    - 'text_image' : 文本 + 图片融合

模型信息：
    - 向量维度：1536
    - 最大序列长度：32768
    - 模型大小：2.21B 参数
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_STACK_LOGGING"] = "0"  # 尝试关闭 macOS malloc 日志
import base64
import mimetypes
from typing import Tuple, List, Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ========= 配置区 =========

# 模型名称或本地路径
MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
# 若已下载到本地，可改为路径，例如：
# MODEL_NAME = "/path/to/gme-Qwen2-VL-2B-Instruct"

# 设备选择：自动检测 CUDA / MPS / CPU
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 数据类型：GPU 用 float16，CPU/MPS 用 float32
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 指数退避算法的基础等待时间（秒）
BASE_BACKOFF = 2.0

# 图片最大体积（本地文件检查），超过则跳过
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB

# 默认指令模板（用于查询端）
DEFAULT_INSTRUCTION = "Find an image that matches the given text."

# ======== 配置区结束 =========


# ========= 全局模型实例 =========

_model = None
_processor = None


def load_model():
    """延迟加载本地模型（单例模式）。

    首次调用时加载模型和处理器，后续调用直接返回已加载的实例。

    Returns:
        Tuple: (model, processor)
    """
    global _model, _processor

    if _model is None:
        print(f"[模型] 正在加载 {MODEL_NAME} ...")
        print(f"[模型] 设备：{DEVICE}，数据类型：{TORCH_DTYPE}")

        # 加载处理器（用于图片预处理）
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,  # 显式声明，消除警告
        )

        # 加载模型（AutoModel 会自动识别为 GmeQwen2VL 类）
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        _model.eval()

        print(f"[模型] 加载完成。向量维度：1536")

    return _model, _processor


# ========= 图片处理工具 =========

def image_to_base64(img_path: str) -> Tuple[str, str]:
    """将本地图片文件转换为 base64 data URI。

    Args:
        img_path: 本地图片路径

    Returns:
        (api_image, store_image)
        - api_image: 形如 "data:image/png;base64,xxxx" 的字符串
        - store_image: 原始图片路径
        失败时返回 ("", "")
    """
    try:
        mime = mimetypes.guess_type(img_path)[0] or "image/png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}", img_path
    except Exception as e:
        print(f"[图片] 本地文件转 base64 失败：{e}")
        return "", ""


def load_image(img: Union[str, Image.Image]) -> Optional[Image.Image]:
    """加载图片为 PIL.Image 对象。

    支持：URL 字符串、本地路径、PIL.Image 对象、base64 data URI。

    Args:
        img: 图片输入

    Returns:
        PIL.Image 对象，失败时返回 None
    """
    if isinstance(img, Image.Image):
        return img

    if not isinstance(img, str) or not img.strip():
        return None

    raw = img.strip()

    # base64 data URI
    if raw.startswith("data:image"):
        try:
            b64_data = raw.split(",", 1)[1]
            from io import BytesIO
            return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")
        except Exception as e:
            print(f"[图片] base64 解码失败：{e}")
            return None

    # URL
    if raw.startswith(("http://", "https://")):
        try:
            from io import BytesIO
            import requests
            resp = requests.get(raw, timeout=10)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[图片] URL 下载失败：{e}")
            return None

    # 本地文件
    if os.path.isfile(raw):
        try:
            return Image.open(raw).convert("RGB")
        except Exception as e:
            print(f"[图片] 本地文件打开失败：{e}")
            return None

    print(f"[图片] 不支持的图片输入：{raw[:80]}...")
    return None


def normalize_image(img: str) -> Tuple[str, str]:
    """规范化图像输入，返回 (api_image, store_image)。

    本地模型可以直接接受 URL、本地路径、PIL.Image 对象，
    这里做基本的有效性检查。

    Args:
        img: 图像路径或 URL

    Returns:
        (api_image, store_image)，无效时返回 ("", "")
    """
    if not img:
        return "", ""

    raw = img.strip()
    low = raw.lower()

    # URL 处理：检查可达性
    if low.startswith(("http://", "https://")):
        try:
            import requests
            head = requests.head(raw, timeout=5, allow_redirects=True)
            if head.status_code != 200:
                print(f"[图片] URL 不可达，status {head.status_code}：{raw}")
                return "", ""
            size = int(head.headers.get("Content-Length") or 0)
            if size and size > MAX_IMAGE_BYTES:
                print(f"[图片] URL 大小 {size} > {MAX_IMAGE_BYTES}，跳过：{raw}")
                return "", ""
        except Exception as e:
            print(f"[图片] HEAD 检查异常：{e}")
        return raw, raw

    # 本地文件
    if os.path.isfile(raw):
        return image_to_base64(raw)

    print(f"[图片] 不支持的类型：{raw[:80]}...")
    return "", ""


# ========= 本地模型推理 =========

def get_text_embedding(
        text: str,
        instruction: Optional[str] = None,
) -> List[float]:
    """获取单条文本的嵌入向量。

    Args:
        text: 文本内容
        instruction: 可选指令（用于查询端），如 "Find an image that matches the given text."

    Returns:
        嵌入向量（list of float，维度 1536）
    """
    model, _ = load_model()

    with torch.no_grad():
        embedding = model.get_text_embeddings(
            texts=[text],
            instruction=instruction,
        )

    return embedding[0].cpu().float().numpy().tolist()


def get_image_embedding(
        image: Union[str, Image.Image],
) -> List[float]:
    """获取单张图片的嵌入向量。

    Args:
        image: 图片输入（URL、本地路径、PIL.Image 对象、base64 data URI）

    Returns:
        嵌入向量（list of float，维度 1536）
    """
    model, _ = load_model()

    pil_image = load_image(image)
    if pil_image is None:
        return []

    with torch.no_grad():
        embedding = model.get_image_embeddings(images=[pil_image])

    return embedding[0].cpu().float().numpy().tolist()


def get_fused_embedding(
        text: str,
        image: Union[str, Image.Image],
) -> List[float]:
    """获取「文本 + 图片」融合嵌入向量。

    Args:
        text: 文本内容
        image: 图片输入（URL、本地路径、PIL.Image 对象、base64 data URI）

    Returns:
        嵌入向量（list of float，维度 1536）
    """
    model, _ = load_model()

    pil_image = load_image(image)
    if pil_image is None:
        return []

    with torch.no_grad():
        embedding = model.get_fused_embeddings(
            texts=[text],
            images=[pil_image],
        )

    return embedding[0].cpu().float().numpy().tolist()


# ========= 核心处理逻辑 =========

def call_local_model(
        input_data: List[Dict],
) -> Tuple[bool, List[float], Optional[int], Optional[float]]:
    """本地模型推理入口，替代原有的 call_dashscope_once()。

    Args:
        input_data: 输入列表，支持三种形式：
            - [{"text": "..."}]                          纯文本
            - [{"image": "..."}]                         纯图片
            - [{"text": "..."}, {"image": "..."}]        图文融合

    Returns:
        (success, embedding, status_code, retry_after)
        - success: 是否成功
        - embedding: 嵌入向量（失败时为空列表）
        - status_code: 模拟 HTTP 状态码（200 / None）
        - retry_after: 本地模型无需重试，始终为 None
    """
    try:
        load_model()

        # 解析输入
        item = input_data[0] if input_data else {}

        #  遍历所有 dict，提取 text 和 image
        text: Optional[str] = None
        image: Optional[str] = None

        for item in input_data:
            if "text" in item and item["text"]:
                text = item["text"]
            if "image" in item and item["image"]:
                image = item["image"]

        #  根据提取到的内容分派到对应方法
        if text and image:
            emb = get_fused_embedding(text, image)
        elif text:
            emb = get_text_embedding(text)
        elif image:
            emb = get_image_embedding(image)
        else:
            print("[本地模型] 输入为空，无法推理")
            return False, [], None, None

        if not emb:
            return False, [], None, None

        return True, emb, 200, None

    except Exception as e:
        print(f"[本地模型] 推理失败：{e}")
        import traceback
        traceback.print_exc()
        return False, [], None, None


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
            # 图文融合：两个独立的 dict
            input_data = [{"text": raw_text}, {"image": api_image}]
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

    # ---- 本地推理 ----
    ok, embedding, status, retry_after = call_local_model(input_data)

    new_item["dense"] = embedding if ok else []
    new_item["_status"] = status
    new_item["_retry_after"] = retry_after

    # 纯图片模式且无文本时，补占位文本
    if mode == "image" and not raw_text:
        new_item["text"] = "图片"

    return new_item


def build_work_items(
        expanded_data: List[Dict],
        combine_text_image: bool = True,
) -> List[Tuple[Dict, str, str]]:
    """构建工作项列表。

    每个工作项为三元组 (item, mode, api_image)。

    Args:
        expanded_data: 原始数据列表
        combine_text_image: 是否将同一 item 的文本+图片合并为一次调用

    Returns:
        List[Tuple]: 工作项列表
    """
    work_items: List[Tuple[Dict, str, str]] = []

    for item in expanded_data:
        content = (item.get("text") or "").strip()
        image_raw = (item.get("image_path") or "").strip()

        # 规范化图片
        api_img, store_img = ("", "")
        if image_raw:
            api_img, store_img = normalize_image(image_raw)

        # 合并模式
        if combine_text_image and content and api_img:
            combo_item = item.copy()
            combo_item["image_path"] = store_img
            work_items.append((combo_item, "text_image", api_img))
            continue

        # 拆分模式
        if content:
            work_items.append((item, "text", ""))

        if api_img:
            pic_item = item.copy()
            pic_item["image_path"] = store_img
            work_items.append((pic_item, "image", api_img))

    return work_items


# ========= 测试入口 =========

if __name__ == "__main__":
    # ---- 测试 1：纯文本 ----
    print("\n" + "=" * 60)
    print("测试 1：纯文本嵌入")
    print("=" * 60)

    ok, emb, _, _ = call_local_model([{"text": "一只在草地上奔跑的金毛犬"}])
    print(f"成功：{ok}，向量维度：{len(emb)}")
    if emb:
        print(f"前 5 个值：{emb[:5]}")

    # ---- 测试 2：纯图片 ----
    print("\n" + "=" * 60)
    print("测试 2：纯图片嵌入")
    print("=" * 60)

    # 替换为你的本地图片路径
    test_image = "/Users/Python/project/project-learn/python-code/Multimodal_RAG/data/Golden_Retriever.jpg"
    if os.path.isfile(test_image):
        ok, emb, _, _ = call_local_model([{"image": test_image}])
        print(f"成功：{ok}，向量维度：{len(emb)}")
    else:
        print(f"跳过：图片不存在 {test_image}")

    # ---- 测试 3：图文融合 ----
    print("\n" + "=" * 60)
    print("测试 3：图文融合嵌入")
    print("=" * 60)

    if os.path.isfile(test_image):
        ok, emb, _, _ = call_local_model([
            {"text": "一只在草地上奔跑的金毛犬"}, {"image": test_image}
        ])
        print(f"成功：{ok}，向量维度：{len(emb)}")
    else:
        print(f"跳过：图片不存在 {test_image}")

    # ---- 测试 4：相似度计算 ----
    print("\n" + "=" * 60)
    print("测试 4：文本-图片相似度")
    print("=" * 60)

    if os.path.isfile(test_image):
        text_emb = get_text_embedding("一只金毛犬")
        img_emb = get_image_embedding(test_image)

        if text_emb and img_emb:
            import numpy as np

            similarity = float(
                np.dot(text_emb, img_emb) /
                (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
            )
            print(f"文本-图片余弦相似度：{similarity:.4f}")
    else:
        print(f"跳过：图片不存在 {test_image}")
