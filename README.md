# Multimodal RAG · 多模态检索增强生成

基于 **视觉 OCR 解析 + 多模态向量化 + Milvus 向量检索** 的多模态 RAG 项目。

从 PDF 文档出发，依次经过版面识别 → Markdown 分割 → 多模态向量化 → 向量库写入，
最终支持 **语义检索、BM25 关键词检索、以图搜图、图文融合、RRF 混合检索** 等多种召回方式，
为后续大模型问答（RAG）提供图文一体的高质量上下文。

## 整体流程

```
PDF 文档
  │  ① dots_ocr 解析（外部 vLLM 服务，模型 dots_ocr，端口 6006）
  ▼
分页 Markdown / JSON / 图片   (output/)
  │  ② MarkdownDirSplitter（按 Header 层级 + 语义切分，提取配图）
  ▼
多模态 Document 列表
  │  ③ doc_to_dict → 多模态向量化（本地 GME / 云端 DashScope 二选一）
  ▼
带稠密向量的结构化数据
  │  ④ write_to_milvus（Milvus 2.4+）
  ▼
Milvus 集合 t_doc_collection（dense 1536 维 + BM25 稀疏向量）
  │  ⑤ MilvusRetriever 多路检索
  ▼
dense_search / sparse_search / image_search /
multimodal_search / hybrid_search（RRF 融合）
```

## 项目结构

```
├── main.py                             # Gradio 交互界面：PDF 上传 → 解析 → 查看 MD 内容
├── dots_ocr/                           # ① OCR 解析：对接外部 vLLM 的 DotsOCR 模型
│   ├── parser.py                       #   PDF/图片 → 分页 md / json / jpg
│   ├── inference.py                    #   vLLM OpenAI 兼容接口调用
│   └── utils/                          #   图片处理、版面坐标后处理、md 格式转换
├── splitters/
│   └── splitters_md.py                 # ② MarkdownDirSplitter：多模态目录分割器
├── milvus_db/                          # ④⑤ Milvus 向量库
│   ├── collections_operator.py         #   集合 Schema（BM25 Function + dense/sparse 索引）
│   ├── db_operator.py                  #   Document → 向量化 → 写入集合
│   └── db_retriever.py                 #   多路检索器 MilvusRetriever
├── utils/                              # ③ Embedding 与通用工具
│   ├── embedding_config.py             #   后端切换（EMBEDDING_BACKEND: local/cloud）
│   ├── embedding_selector.py           #   按配置选择本地或云端实现
│   ├── gme_qwen2_vl_2b_embedding.py    #   本地 Alibaba-NLP/gme-Qwen2-VL-2B-Instruct（1536 维）
│   ├── multimodal_embedding.py         #   云端 DashScope multimodal-embedding-v1（限流/429重试）
│   ├── env_utils.py                    #   读取 .env，导出 API Key / Milvus 配置
│   ├── common_utils.py                 #   文件路径等通用函数
│   └── log_utils.py                    #   loguru 日志
├── embedding_demo/
│   ├── download_model_embedding.py     #   下载本地多模态模型
│   └── custom_embedding.py             #   LangChain / Ragas 双接口嵌入集成示例
├── evaluate/                           # RAG 效果评估（Ragas 指标）
│   ├── evaluate_single_turn.py         #   单轮评估：上下文相关性 / 答案相关性 / 精确度
│   └── evaluate_multi_turn.py          #   多轮评估：目标达成度 / 主题一致性
├── models/
│   └── init_chat_model_llm.py          #   LLM 客户端统一初始化（GLM / DeepSeek / ZhipuAI）
├── data/                               # 输入数据（示例 PDF、以图搜图测试图片）
├── output/                             # OCR 解析结果（每页 md / json / jpg）
├── images/                             # 分割器提取出的配图（不入库）
├── .env.example                        # 环境变量模板（复制为 .env 使用）
├── requirements.txt                    # 依赖
└── venv.txt                            # conda 环境创建指引
```

## 核心模块说明

### 1. OCR 解析（dots_ocr）
对接外部部署的多模态 OCR 模型（DotsOCR，OpenAI 兼容接口，默认 `127.0.0.1:6006`），
把 PDF 按页解析出 **Markdown 文本 + 版面 JSON + 页面图片**，输出到 `output/<文档名>/`。
图片会做超分/压缩（`min_pixels` / `max_pixels`）、版面坐标预处理与 md 结构整理。

### 2. Markdown 目录分割器（MarkdownDirSplitter）
- 按 **Header 1~N 层级** 组织文档结构，先粗切再语义切分（`SemanticChunker`）。
- 从 md 中**提取配图**到 `images/`，生成两类 Document：
  - `embedding_type="text"`：纯文本片段（标题已拼接进正文）
  - `embedding_type="image"`：图片片段（`page_content` 为图片路径）

### 3. 多模态向量化（utils 双后端）
通过环境变量 `EMBEDDING_BACKEND` 一键切换：

| 后端 | 模型 | 向量维度 | 说明 |
|---|---|---|---|
| `local` | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | 1536 | 本地推理，需独立 conda 环境，无限流 |
| `cloud` | DashScope `multimodal-embedding-v1` | 由服务端决定 | 内置 RPM 限流与 429 指数退避重试 |

统一支持三种输入模式：纯文本（`text`）、纯图片（`image`）、图文融合（`text_image`）。
上层 `db_operator` 只依赖 `embedding_selector`，不感知具体实现。

### 4. Milvus 向量库（milvus_db）
- **集合 Schema**：主键自增 + `text`（jieba 分词 BM25）+ `category` / `filename` / `filetype` / `image_path` / `title`
  + `sparse`（BM25 函数自动生成）+ `dense`（1536 维，IP 相似度，AUTOINDEX）。
- **写入**：`do_save_to_milvus` 完成 Document → 字典 → 向量化 → 批量写入，
  云端口径支持 429 重试，失败数据自动过滤并记录日志。
- **检索**（`MilvusRetriever`）：

| 方法 | 用途 |
|---|---|
| `dense_search` | 语义相似检索（1536 维向量） |
| `sparse_search` | BM25 关键词全文检索 |
| `image_search` | 以图搜图 / 以图搜文（可选 `only_image`） |
| `multimodal_search` | 文本 + 图片融合为一向量再检索 |
| `hybrid_search` | 客户端 RRF 融合（两路都命中可加权） |
| `hybrid_search_native` | 服务端原生 RRF 融合（Milvus 2.4+） |

### 5. RAG 效果评估（evaluate）
基于 Ragas 对 RAG 链路做量化评估，入口在 `evaluate/`：

| 脚本 | 指标 | 说明 |
|---|---|---|
| `evaluate_single_turn.py` | 上下文相关性 / 答案相关性 / 上下文精确度 | 上下文精确度支持有/无参考答案两种模式 |
| `evaluate_multi_turn.py` | 目标达成度 / 主题一致性（f1 / precision / recall） | 目标达成度为 `None` 时标记「评估失败（输出截断）」 |

- **评估 LLM**：GLM（`glm-5.3-flash`，异步 OpenAI 兼容客户端，`max_tokens=4096`）。
- **评估嵌入**：`ModernQwen2Embeddings`（`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`，直接实现 Ragas `BaseRagasEmbedding` 接口，见 `embedding_demo/custom_embedding.py`）。

### 6. 多模态嵌入集成示例（embedding_demo）
`custom_embedding.py` 提供两个可直接复用的嵌入类：
- `CustomQwen3Embeddings`：Qwen3 文本嵌入，适配 **LangChain** `Embeddings` 接口。
- `ModernQwen2Embeddings`：gme-Qwen2-VL 多模态嵌入，适配 **Ragas** `BaseRagasEmbedding` 接口（含 `embed_text` / `embed_texts` 及对应异步方法）。

## 环境准备

### 1. 创建 conda 环境

推荐两个独立环境（本地 GME 模型与主环境存在依赖冲突，原因见下文「两个依赖文件说明」）：

```bash
# 主环境
conda create -n Multimodal_RAG python=3.11 -y
conda activate Multimodal_RAG
pip install -r requirements_multimodal_rag.txt

# 本地 GME 模型专用环境
conda create -n gme_qwen_local python=3.11 -y
conda activate gme_qwen_local
pip install -r requirements_gme_qwen_local.txt
# 注：该文件已锁定 transformers==4.51.3 / huggingface-hub==0.36.2 / tokenizers==0.21.4
```

> 更简明的步骤清单见 `venv.txt`。

#### 两个依赖文件说明（requirements_*.txt）

| 文件 | 对应环境 | 包数 | 用途 |
|---|---|---|---|
| `requirements_multimodal_rag.txt` | `Multimodal_RAG`（主环境） | 125 | Gradio 界面 / 检索 / OCR 主链路，现代 HF 依赖栈 |
| `requirements_gme_qwen_local.txt` | `gme_qwen_local`（本地 GME） | 145 | gme-Qwen2-VL 本地推理 + Ragas 评估，锁定兼容依赖栈 |

两个环境是**刻意分离**的：`gme_qwen_local` 在覆盖主环境全部 125 个包之外，还多出 Ragas 评估链路所需的 `ragas / datasets / accelerate / scikit-network` 等 20 个包；但两环境共有包中有 17 个**版本不同**，核心差异集中在 HF / Transformers 栈：

| 包 | `Multimodal_RAG`（主） | `gme_qwen_local` | 说明 |
|---|---|---|---|
| transformers | 5.17.0 | 4.51.3 | gme-VL 模型必须锁定旧版才能加载推理 |
| tokenizers | 0.23.2 | 0.21.4 | 随 transformers 降级 |
| huggingface_hub | 1.31.0 | 0.36.2 | 随 transformers 降级 |
| sentence-transformers | 6.0.1 | 5.3.0 | 本地多模态嵌入所用版本 |
| openai | 3.13.0 | 1.109.1 | 主环境新版 SDK / gme 环境旧版 |
| gradio | 6.26.0 | 6.27.0 | 轻微差异 |

> ⚠️ **混装即冲突**：`transformers 5.17` 会导致 `gme-Qwen2-VL-2B-Instruct` 本地推理失败；反之用 gme 清单装主环境会引入旧版 `openai` 等。重建环境时请**严格按各自文件安装**，并保持 `python=3.11`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

在 `.env` 中按需填写：

| 变量 | 说明 |
|---|---|
| `EMBEDDING_BACKEND` | `local`（本地 GME）/ `cloud`（DashScope） |
| `ALIBABA_API_KEY` | 选用 `cloud` 后端时必填 |
| `MILVUS_URI` | Milvus 地址，默认 `http://127.0.0.1:19530` |
| `MILVUS_COLLECTION_NAME` | 集合名，默认 `t_doc_collection` |
| `QWEN3_VL_EMBEDDING_PATH` | 本地模型快照路径（可选，默认读 HuggingFace 缓存） |
| `ZHIPU_API_KEY` | 智谱 AI 密钥（用于 `init_chat_model_llm.py` 中的 ZhipuAI 客户端） |

> `.env` 已被 `.gitignore` 排除，**严禁提交**，只提交 `.env.example` 模板。

### 3. 启动外部依赖

- **Milvus**：本地 Docker 部署或 Zilliz Cloud，保持与 `MILVUS_URI` 一致。
- **DotsOCR 服务**：自行部署 DotsOCR 并用 vLLM 以 OpenAI 兼容方式启动（默认 `127.0.0.1:6006`）。
- **本地模型**：切换 `EMBEDDING_BACKEND=local` 后，先下载模型：

```bash
python embedding_demo/download_model_embedding.py   # 需在 gme_qwen_local 环境
```

## 运行与使用

### 初始化向量集合

```bash
python milvus_db/collections_operator.py
```

### 构建知识库（从 Markdown → 向量库）

```bash
python splitters/splitters_md.py
```

> 入口中指定了 `md_dir` 与 `images_output_dir`，执行后完成分割、向量化并写入 Milvus。

### 启动 Gradio 交互界面

```bash
python main.py
```

流程：上传 PDF → 点击「解析PDF」→ 下拉框选择查看每页 MD 内容（「存入知识库」为规划中的下一步，目前入库走 `splitters_md.py`）。

### 检索测试

```bash
python milvus_db/db_retriever.py
```

内置演示：文本语义检索、BM25 关键词检索、RRF 混合检索、以图搜图、图文融合检索。

### RAG 效果评估

```bash
python evaluate/evaluate_single_turn.py   # 单轮评估
python evaluate/evaluate_multi_turn.py    # 多轮评估
```

## 目录数据约定

| 目录 | 是否入库 | 说明 |
|---|---|---|
| `data/` | ✔ | 输入数据（PDF、示例图片） |
| `output/` | ✔ | OCR 解析结果（每页 md / json / jpg） |
| `images/` | ✘ | 分割器产出的配图（可由代码重新生成） |
| `logs/` | ✘ | 运行日志 |
| `.env` | ✘ | 密钥等本地配置，见 `.env.example` |

## Tech Stack

- **解析**：DotsOCR（vLLM）· PyMuPDF
- **分割**：LangChain（MarkdownHeaderTextSplitter / SemanticChunker）
- **向量化**：gme-Qwen2-VL-2B-Instruct（本地）· DashScope multimodal-embedding-v1（云端）
- **向量库**：Milvus 2.4+（BM25 稀疏索引 + AUTOINDEX / IP / RRF）
- **界面**：Gradio