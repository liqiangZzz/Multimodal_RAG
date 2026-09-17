import os
import random
import time
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pymilvus import MilvusException

from milvus_db.collections_operator import client, MILVUS_COLLECTION_NAME
from models.init_chat_model_llm import glm_llm_flash
from utils.common_utils import get_surrounding_text_content
# 统一从 selector 导入，不再直接依赖具体实现
from utils.embedding_selector import (
    build_work_items,
    process_item_with_guard,
    ENABLE_RETRY,
    RETRY_ON_429,
    MAX_429_RETRIES,
    BASE_BACKOFF,
)
from utils.gme_qwen2_vl_2b_embedding import image_to_base64
from utils.log_utils import log


# =============================================================================
# Document -> Dict
# =============================================================================

def doc_to_dict(docs: List[Document]) -> List[Dict]:
    """ Document 列表转换为字典列表。

     提取字段：
        - text        : 文本内容（仅 embedding_type == "text" 时有效）
        - category    : embedding_type
        - filename    : source
        - filetype    : source 的后缀
        - image_path  : 图片路径（仅 embedding_type == "image" 时有效）
        - title       : 拼接所有 Header 层级的标题
    Args:
        docs:Document列表

    Returns:
        字典Document列表
    """

    result_list = []

    for doc in docs:
        # 初始化一个空字典来存储当前文档的信息
        doc_dict = {}
        # 提取文档的元数据
        metadata = doc.metadata

        # 1. 提取 text 内容（仅文档 embedding_type 为 text 时有效）
        if metadata.get("embedding_type") == "text":
            doc_dict["text"] = doc.page_content
        else:
            # 或者设置为空字符串 ''，根据需要调整
            doc_dict["text"] = None

        # 2.提取 category (embedding_type)
        doc_dict['category'] = metadata.get('embedding_type', '')

        # 3. 提取 filename  (source)
        source = metadata.get('source', '')
        doc_dict['filename'] = source

        # 4.提取 filetype  (source 中文件名的后缀)
        _, file_extension = os.path.splitext(source)
        doc_dict['filetype'] = file_extension.lower()  # 转换为小写,如 '.jpg'

        # 5. 提取 image_path (仅当 embedding_type 为 image 时有效)
        if metadata.get("embedding_type") == "image":
            doc_dict['image_path'] = doc.page_content
        else:
            doc_dict['image_path'] = None  # 或者设置为空字符串 ''，根据需要调整

        # 6. 提取 title （拼接所有 Header 层级的标题）
        headers = []

        # 假设 Header 的键可能为 'Header 1', 'Header 2', 'Header 3' 等，我们按层级顺序拼接
        # 我们需要先收集所有存在的 Header 键，并按层级排序
        header_keys = [
            key for key in metadata.keys() if key.startswith("Header")
        ]
        # 按 Header 后的数字排序，确保层级顺序
        header_keys_sorted = sorted(
            header_keys,
            key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else x,
        )

        for header_key in header_keys_sorted:
            value = metadata.get(header_key, '').strip()
            if value:  # 只添加非空值
                headers.append(value)

        # 将所有的非空的 Header 值用字符串或空格连接起来
        doc_dict['title'] = ' --> '.join(headers) if headers else ''  # 你也可以用其他连接符，如空格

        # 7. 如果是文本文档，将标题和文本拼接起来
        if not doc_dict['image_path']:
            if doc_dict['title']:
                doc_dict['text'] = doc_dict['title'] + ' ：' + (doc_dict['text'] or '')
            else:
                doc_dict['text'] = doc_dict['text'] or ''
        # 将当前文档的字典添加到结果列表中
        result_list.append(doc_dict)

    return result_list


# =============================================================================
# 写入 Milvus
# =============================================================================
def write_to_milvus(processed_data: List[Dict]):
    """
    将处理后的数据写入 Milvus 数据库
    Args:
        processed_data: 处理后的数据列表
    """

    if not processed_data:
        print("[Milvus] 没有可写入的数据。")
        return

    # ---- 插入前清理内部字段 ----
    # Milvus schema 中未定义的字段不能插入。
    # _status / _retry_after 是 process_item_with_guard 内部使用的，
    # 写库前必须删除。
    internal_keys = {"_status", "_retry_after"}
    cleaned_data = []
    for item in processed_data:
        cleaned = {k: v for k, v in item.items() if k not in internal_keys}
        cleaned_data.append(cleaned)

    try:
        insert_result = client.insert(collection_name=MILVUS_COLLECTION_NAME, data=cleaned_data)
        print(f"[Milvus] 成功插入 {insert_result['insert_count']} 条记录。IDs 示例: {insert_result['ids'][:5]}")
    except MilvusException as e:
        print(f"[Milvus] 插入失败: {e}")
        log.exception("Milvus 插入失败")


# =============================================================================
# 重试退避计算（仅云端模式使用）
# =============================================================================
def _calc_backoff(attempts: int, retry_after: Optional[float]) -> float:
    """计算本次重试的等待时间。

    优先使用服务端返回的 Retry-After，否则用指数退避 + jitter。
    """
    if retry_after is not None:
        return min(retry_after, 60.0)

    exponential = BASE_BACKOFF * (2 ** (attempts - 1))
    jitter = random.uniform(0, exponential * 0.2)
    return min(exponential + jitter, 60.0)


def generate_image_description(data_list):
    """
    处理文档数据，为每个图片字典生成多模态描述
    Args:
        data_list: 包含字典的列表

    Returns:
        包含完整结果的新列表
    """

    for index, item in enumerate(data_list):
        if item.get("image_path"):  # 检查是否为图片字典
            # 获取前后文本内容
            prev_text, next_text = get_surrounding_text_content(data_list, index)

            # 将图片转换成 base64
            image_data = image_to_base64(item.get("image_path"))[0]

            # 构建上下文提示词
            if prev_text and next_text:
                context_prompt = (
                    f"前文内容: {prev_text}\n"
                    f"后文内容: {next_text}\n\n"
                    "请根据以上上下文和图片内容，生成对该图片的简洁描述，"
                    "描述内容长度最好不超过300个汉字。\n"
                    "注意：图片可能与前文、后文或两者都相关，请综合分析。"
                )
            elif prev_text:
                context_prompt = (
                    f"前文内容: {prev_text}\n\n"
                    "请根据以上上下文和图片内容，生成对该图片的简洁描述，"
                    "描述内容长度最好不超过300个汉字。"
                )
            elif next_text:
                context_prompt = (
                    f"后文内容: {next_text}\n\n"
                    "请根据以上上下文和图片内容，生成对该图片的简洁描述，"
                    "描述内容长度最好不超过300个汉字。"
                )
            else:
                context_prompt = (
                    "请描述这张图片的内容，生成对该图片的简洁描述，"
                    "描述内容长度最好不超过300个汉字。"
                )

            # 构建多模态消息
            message = HumanMessage(
                content=[
                    {"type": "text", "text": context_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ]
            )

            # 调用模型生成描述
            try:
                response = glm_llm_flash.invoke([message])
                item["text"] = response.content.strip()
            except Exception as e:
                log.exception(f"[图片描述] 第 {index} 张图片生成描述失败")
                item["text"] = ""
    return data_list


# =============================================================================
# 核心处理流程
# =============================================================================
def do_save_to_milvus(docs: List[Document]) -> List[Dict]:
    """把 Document 列表向量化后写入 Milvus。

    流程：
        docs -> doc_to_dict -> generate_image_description -> build_work_items
             -> process_item_with_guard -> 过滤空向量 -> write_to_milvus

    重试：
    - local : 不重试，失败直接跳过。
    - cloud : 仅对 _status == 429 重试，其他失败跳过。

    Args:
        docs: Document 列表

    Returns:
        带向量的字典列表（仅包含成功生成向量的数据）
    """
    # ---- 第一步：Document -> Dict ----
    # 把Splitter之后的的数据（document对象列表），先转换为字典
    data_dicts = doc_to_dict(docs)
    log.info(f"[Milvus] 转换后字典数量: {len(data_dicts)}")

    # ---- 第二步：为图片生成描述（写入 description，不覆盖 text）----
    expanded_data = generate_image_description(data_dicts)
    log.info(f"[Milvus] 图片描述生成完成，共 {len(expanded_data)} 条")

    # ---- 第三步：构建工作项 ----
    work_items = build_work_items(expanded_data)
    log.info(f"[Milvus] 构建工作项数量: {len(work_items)}")

    if not work_items:
        log.warning("[Milvus] 没有可处理的工作项，直接退出")
        return []

    # ---- 第四步：向量化 ----
    embedded_data: List[Dict] = []
    total = len(work_items)

    for index, (item, mode, api_img) in enumerate(work_items, start=1):

        # -------- 本地模式：单次处理 --------
        if not ENABLE_RETRY:
            try:
                result = process_item_with_guard(
                    item.copy(), mode=mode, api_image=api_img
                )
            except Exception:
                log.exception(f"[异常] 处理第 {index} 项失败，跳过")
                continue

            if result.get("dense"):
                embedded_data.append(result)
            else:
                log.warning(f"[跳过] idx={index}, mode={mode} 无向量")

            if index % 20 == 0 or index == total:
                log.info(f"[进度] 已处理 {index}/{total}")
            continue

        # -------- 云端模式：429 重试 --------
        attempts = 0
        while True:
            try:
                result = process_item_with_guard(
                    item.copy(), mode=mode, api_image=api_img
                )
            except Exception:
                log.exception(f"[异常] 处理第 {index} 项失败，跳过")
                break

            if result.get("dense"):
                embedded_data.append(result)
                break

            status = result.get("_status")
            retry_after = result.get("_retry_after")

            if status != 429:
                log.warning(
                    f"[跳过] idx={index}, mode={mode}, status={status}, "
                    f"text={str(item.get('text', ''))[:50]!r}"
                )
                embedded_data.append(result)
                break

            if not RETRY_ON_429 or attempts >= MAX_429_RETRIES:
                log.warning(
                    f"[429重试] idx={index}, mode={mode}, "
                    f"已达最大重试次数 {MAX_429_RETRIES}，放弃"
                )
                embedded_data.append(result)
                break

            # 可以重试，等待 retry_after 秒后重试一次
            attempts += 1
            backoff = _calc_backoff(attempts, retry_after)
            log.info(
                f"[429重试] idx={index}, mode={mode}, "
                f"第 {attempts}/{MAX_429_RETRIES} 次，等待 {backoff:.2f}s"
            )
            time.sleep(backoff)
            # while True 回到顶部重新处理

        if index % 20 == 0 or index == total:
            log.info(f"[进度] 已处理 {index}/{total}")

    # ---- 第五步：过滤空向量 ----
    valid_data = [d for d in embedded_data if d.get("dense")]
    if len(valid_data) < len(embedded_data):
        log.warning(
            f"[Milvus] 过滤掉 {len(embedded_data) - len(valid_data)} 条无向量数据"
        )

    # ---- 第六步：写入 Milvus ----
    write_to_milvus(valid_data)
    return valid_data