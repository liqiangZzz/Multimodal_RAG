"""
Embedding 后端选择器。

根据 EMBEDDING_BACKEND 决定使用哪个模块：
    - local -> gme_qwen2_vl_2b_embedding（本地 GME 模型）
    - cloud -> multimodal_embedding（DashScope 云端）

设计原则：
    限流、重试、异常分类、字段透传，全部由具体 embedding 模块内部负责。
    本 selector 只做后端选择，不向外暴露 limiter。
    上层 db_operator 只 import 本模块，不感知底层实现。

统一暴露的接口：
    - build_work_items(expanded_data)
    - process_item_with_guard(item, mode, api_image)
    - ENABLE_RETRY : bool  是否需要在调用方重试（仅云端为 True）
    - RETRY_ON_429 : bool
    - MAX_429_RETRIES : int
    - BASE_BACKOFF : float
"""

from utils.embedding_config import EMBEDDING_BACKEND


if EMBEDDING_BACKEND == "cloud":
    # ---- 云端 DashScope ----
    # 限流在 multimodal_embedding.call_dashscope_once() 内部完成。
    # 调用方无需也不应再调用 limiter.acquire()。
    from utils.multimodal_embedding import (
        build_work_items,
        process_item_with_guard,
        RETRY_ON_429,
        MAX_429_RETRIES,
        BASE_BACKOFF,
    )

    ENABLE_RETRY = True

elif EMBEDDING_BACKEND == "local":
    # ---- 本地 GME ----
    from utils.gme_qwen2_vl_2b_embedding import (
        build_work_items,
        process_item_with_guard,
    )

    # 本地模型没有 429，不需要重试
    RETRY_ON_429 = False
    MAX_429_RETRIES = 0
    BASE_BACKOFF = 1.0
    ENABLE_RETRY = False

else:
    # embedding_config 已拦截非法值，这里只是兜底
    raise RuntimeError(f"不支持的 EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


__all__ = [
    "build_work_items",
    "process_item_with_guard",
    "ENABLE_RETRY",
    "RETRY_ON_429",
    "MAX_429_RETRIES",
    "BASE_BACKOFF",
]