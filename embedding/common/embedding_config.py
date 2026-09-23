"""
Embedding 后端配置。

通过环境变量 EMBEDDING_BACKEND 切换：
    - "local" : 使用本地 Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
    - "cloud" : 使用 DashScope multimodal-embedding-v1

默认值可以在 EMBEDDING_BACKEND_DEFAULT 中修改。
"""

import os

# 默认后端：local 或 cloud
EMBEDDING_BACKEND_DEFAULT = "local"

# 优先读环境变量，否则用默认值
EMBEDDING_BACKEND = os.getenv(
    "EMBEDDING_BACKEND", EMBEDDING_BACKEND_DEFAULT
).strip().lower()

if EMBEDDING_BACKEND not in ("local", "cloud"):
    raise ValueError(
        f"EMBEDDING_BACKEND 必须是 'local' 或 'cloud'，当前值：{EMBEDDING_BACKEND!r}"
    )