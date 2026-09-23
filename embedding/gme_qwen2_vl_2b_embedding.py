"""
多模态嵌入处理模块（本地部署版）

使用本地部署的 Alibaba-NLP/gme-Qwen2-VL-2B-Instruct 模型，
替代原有的 DashScope 云端 multimodal-embedding-v1 API。

支持三种输入模式：
    - 'text'       : 纯文本
    - 'image'      : 纯图片
    - 'text_image' : 文本 + 图片融合

模型信息：
    - 向量维度：1536（已做 L2 归一化）
    - 模型大小：2.21B 参数

全项目只有本模块创建模型实例（进程内单例），
`embedding.custom_embedding` 的 ModernQwen2Embeddings 复用这里的实例，
因此语义切分 / 向量化 / 检索 / 评估共用同一份权重，不会重复加载。
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_STACK_LOGGING"] = "0"  # 尝试关闭 macOS malloc 日志
import base64
import mimetypes
from io import BytesIO
from typing import Tuple, List, Dict, Optional, Union

from PIL import Image
from sentence_transformers import SentenceTransformer

# ========= 配置区 =========

# 模型名称或本地路径
MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
# 若已下载到本地，可改为路径，例如：
# MODEL_NAME = "/path/to/gme-Qwen2-VL-2B-Instruct"

# 指数退避算法的基础等待时间（秒）
BASE_BACKOFF = 2.0

# 图片最大体积（本地文件检查），超过则跳过
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB

# ======== 配置区结束 =========


# ========= 全局模型实例（进程内单例，全项目共享） =========

_models: Dict[str, SentenceTransformer] = {}


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """加载本地 GME 模型（进程内单例，全项目共用同一份）。

    统一走 sentence-transformers 官方接口：
      - trust_remote_code=True 启用模型自带的 MultiModalTransformer；
      - local_files_only=True 只读本地缓存，避免联网校验拖慢启动；
      - device=None 由 sentence-transformers 自动选择 cuda / mps / cpu。

    模型自带的 processor 会按 config 的 min/max_image_tokens（256 / 1280）构建，
    与官方示例一致，因此这里不需要再单独建 AutoProcessor。

    Args:
        model_name: 模型名或本地路径

    Returns:
        SentenceTransformer: 共享的模型实例（重复调用只加载一次）
    """
    if model_name not in _models:
        print(f"[模型] 正在加载 {model_name} ...")
        _models[model_name] = SentenceTransformer(
            model_name,
            local_files_only=True,
            trust_remote_code=True,
            device=None,  # 自动选择 cuda / mps / cpu
        )
        model = _models[model_name]
        print(f"[模型] 加载完成。设备：{model.device}，向量维度：1536")

    return _models[model_name]


def encode_inputs(inputs: List[Dict]) -> List[List[float]]:
    """通用编码入口（本模块与 ModernQwen2Embeddings 共用，保证只有一份模型）。

    Args:
        inputs: gme 要求的 dict 列表，每项形如：
                - {"text": "..."}                    纯文本
                - {"image": <PIL/路径/URL/base64>}    纯图片
                - {"text": "...", "image": ...}      图文融合
                可选 {"prompt": "..."} 指定指令（不传则用模型默认指令）

    Returns:
        List[List[float]]: 每项的 1536 维 L2 归一化向量
    """
    embeddings = load_model().encode(
        inputs,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    # numpy array 直接 .tolist()
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()

    # torch tensor 逐条转换
    return [
        e.cpu().tolist() if hasattr(e, "cpu") else list(e)
        for e in embeddings
    ]


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

def image_to_data_uri(image: Union[str, Image.Image]) -> str:
    """把图像统一成字符串形式的引用。

    sentence-transformers 的 encode 会先按 len() 对输入排序，而 PIL.Image 没有 len()，
    直接传 PIL 会报 `TypeError: object of type 'Image' has no len()`；
    因此传给模型前统一用字符串：原本就是字符串（URL / 本地路径 / data URI）直接返回，
    PIL.Image 则转成 base64 data URI（模型的 fetch_image 原生支持这种写法）。

    Args:
        image: PIL.Image 或字符串形式的图像引用

    Returns:
        str: 可直接交给模型的图像引用字符串
    """
    if isinstance(image, str):
        return image

    buf = BytesIO()
    image.save(buf, format="PNG")  # PNG 无损，与直接传 PIL 的像素一致
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def get_text_embedding(
        text: str,
        instruction: Optional[str] = None,
) -> List[float]:
    """获取单条文本的嵌入向量。

    Args:
        text: 文本内容
        instruction: 可选指令（查询端）。当前调用方均不传，
            未传时使用模型默认指令（"You are a helpful assistant."）；
            如需指定，会作为 dict 的 prompt 键交给模型。

    Returns:
        嵌入向量（list of float，维度 1536）
    """
    item: Dict = {"text": text}
    if instruction:
        item["prompt"] = instruction
    return encode_inputs([item])[0]


def get_image_embedding(
        image: Union[str, Image.Image],
) -> List[float]:
    """获取单张图片的嵌入向量。

    Args:
        image: 图片输入（URL、本地路径、PIL.Image 对象、base64 data URI）

    Returns:
        嵌入向量（list of float，维度 1536），图片无效时返回 []
    """
    pil_image = load_image(image)
    if pil_image is None:
        return []

    return encode_inputs([{"image": image_to_data_uri(pil_image)}])[0]


def get_fused_embedding(
        text: str,
        image: Union[str, Image.Image],
) -> List[float]:
    """获取「文本 + 图片」融合嵌入向量。

    Args:
        text: 文本内容
        image: 图片输入（URL、本地路径、PIL.Image 对象、base64 data URI）

    Returns:
        嵌入向量（list of float，维度 1536），图片无效时返回 []
    """
    pil_image = load_image(image)
    if pil_image is None:
        return []

    return encode_inputs([{"text": text, "image": image_to_data_uri(pil_image)}])[0]


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
