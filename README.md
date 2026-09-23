# Multimodal RAG · 多模态检索增强生成

基于 **视觉 OCR 解析 + 多模态向量化 + Milvus 向量检索** 的多模态 RAG 项目。

从 PDF 文档出发，依次经过版面识别 → Markdown 分割 → 多模态向量化 → 向量库写入，
最终支持 **语义检索、BM25 关键词检索、以图搜图、图文融合、RRF 混合检索** 等多种召回方式，
为后续大模型问答（RAG）提供图文一体的高质量上下文。

项目包含两条链路，**使用不同的向量集合，互不影响**：

| 链路 | 入口 | 向量集合 | 用途 |
|---|---|---|---|
| 文档知识库 | `main.py` / `splitters/splitters_md.py` | `t_doc_collection` | PDF → OCR → 分割 → 向量化 → 入库，面向文档检索 |
| 多轮对话 RAG 工作流 | `python -m graph.workflow[_gradio]` | `t_context_collection` | 基于该用户的历史问答作答，**本地查不到自动联网**并写回 |

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

多轮对话 RAG 工作流（`graph/`）是另一条链路，用 LangGraph 编排、独立于上面的文档链路：

```
用户提问（文本 / 图片 / 图文）
  │  process_input 判别输入类型
  ▼
first_chatbot（LLM 调用 search_context 工具）
  │  search_context：文本 → gme 嵌入 → 按 username 过滤 → hybrid_search（RRF）
  ▼
上下文相关性评分 ≥ 0.5 ？
  ├── 是 → second_chatbot 基于历史对话上下文作答
  └── 否 → web_search_node 联网检索 → fourth_chatbot 基于搜索结果作答
  ▼
evaluate_node（答案相关性）
  ├── ≥ 0.7 → END，答案自动写回 t_context_collection
  └── < 0.7 → 人工审批：approve → END ／ rejected → 联网重新作答

（仅输入图片时走另一条支线：process_input → retriever_node → third_chatbot → …）
```

## 嵌入模型

- 全链路只用一个嵌入模型：**`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`**（**1536 维**），
  同时用于**语义切分、文档向量化、检索与 RAG 评估**，四个环节共享同一向量空间。
- `embedding/custom_embedding.py` 的 `ModernQwen2Embeddings` 双继承
  **`Embeddings`（LangChain）+ `BaseRagasEmbedding`（Ragas）**，只做协议转换，不持有也不加载模型。

| 环节 | 代码入口 | 调用接口 |
|---|---|---|
| 语义切分 | `splitters/splitters_md.py` | LangChain `Embeddings`（`embed_query` / `embed_documents`） |
| 文档向量化 | `embedding/gme_qwen2_vl_2b_embedding.py` | 本地 `sentence-transformers`（`text` / `image` / `text_image`） |
| 检索 | `milvus_db/db_retriever.py` | 同上，查询向量与库内向量同空间 |
| RAG 评估 | `evaluate/evaluate_*.py` | Ragas `BaseRagasEmbedding`（`embed_text` / `embed_texts`） |

> 模型实例由 `embedding/gme_qwen2_vl_2b_embedding.py` 以进程内单例持有（`load_model()`），
> `ModernQwen2Embeddings` 复用该实例，因此**全项目只加载一份 2B 权重**。
> 适配层只依赖基座的 `encode_inputs`，没有 `__init__`，**实例化不触发加载**，
> 首次编码时才懒加载（约 15~20 秒，属正常现象）。
> 该模型带自定义模块，必须在 `transformers==4.51.3` 下加载（依赖已锁版，见 `venv.txt`）。
> Milvus 集合的 `dense` 字段维度 **1536** 即由该模型决定，**换模型必须重建集合**。

## 项目结构

```
├── main.py                             # Gradio 交互界面：PDF 上传 → 解析 → 查看 MD → 存入知识库
├── dots_ocr/                           # ① OCR 解析：对接外部 vLLM 的 DotsOCR 模型
│   ├── parser.py                       #   PDF/图片 → 分页 md / json / jpg
│   ├── inference.py                    #   vLLM OpenAI 兼容接口调用
│   └── utils/                          #   图片处理、版面坐标后处理、md 格式转换
├── splitters/
│   └── splitters_md.py                 # ② MarkdownDirSplitter：多模态目录分割器
├── embedding/                          # ③ 多模态嵌入（全链路统一入口）
│   ├── gme_qwen2_vl_2b_embedding.py    #   本地 GME 基座：模型单例 + 编码 + 图片工具
│   ├── custom_embedding.py             #   LangChain / Ragas 双接口适配（复用基座单例）
│   ├── multimodal_embedding.py         #   云端 DashScope multimodal-embedding-v1（限流/429重试）
│   └── common/
│       ├── embedding_config.py         #   后端切换（EMBEDDING_BACKEND: local/cloud）
│       ├── embedding_selector.py       #   按配置选择本地或云端实现
│       └── download_model_embedding.py #   下载本地多模态模型
├── milvus_db/                          # ④⑤ Milvus 向量库
│   ├── collections_operator.py         #   集合 Schema（BM25 Function + dense/sparse 索引）
│   ├── db_operator.py                  #   Document → 向量化 → 写入集合
│   └── db_retriever.py                 #   多路检索器 MilvusRetriever
├── utils/                              #   通用工具
│   ├── env_utils.py                    #   读取 .env，导出 API Key / Milvus 配置
│   ├── common_utils.py                 #   文件路径等通用函数
│   └── log_utils.py                    #   loguru 日志
├── evaluate/                           # RAG 效果评估（Ragas 指标）
│   ├── evaluate_single_turn.py         #   单轮评估：上下文相关性 / 答案相关性 / 精确度
│   └── evaluate_multi_turn.py          #   多轮评估：目标达成度 / 主题一致性
├── graph/                              # ⑥ 多轮对话 RAG 工作流（LangGraph，独立于文档链路）
│   ├── workflow.py                     #   命令行入口：python -m graph.workflow
│   ├── workflow_gradio.py              #   Gradio 界面入口：python -m graph.workflow_gradio
│   ├── custom_state.py                 #   图状态：输入类型 / 检索结果 / 评估分数 / 用户名
│   ├── all_router.py                   #   全部条件路由函数
│   ├── tools.py                        #   search_context（查历史对话）/ my_search（智谱联网）
│   ├── search_node.py                  #   自定义上下文检索工具节点 + retriever_node
│   ├── evaluate_node.py                #   答案相关性评估节点
│   ├── save_context.py                 #   答案写回上下文库（含写库质量闸门）
│   ├── print_messages.py               #   运行时消息打印辅助
│   └── graph_db/                       #   graph 专属向量库代码（不与根 milvus_db 同名）
│       ├── collections_operator_graph.py  # t_context_collection 集合定义（BM25 + dense）
│       └── db_retriever_graph.py          # MilvusRetriever 多路混合检索器
├── models/
│   └── init_chat_model_llm.py          #   LLM 客户端统一初始化（GLM / DeepSeek / ZhipuAI）
├── data/                               # 输入数据（示例 PDF、以图搜图测试图片）
├── output/                             # OCR 解析结果（每页 md / json / jpg）
│   └── images/                         #   分割器提取出的配图
├── .env.example                        # 环境变量模板（复制为 .env 使用）
├── requirements.txt                    # 依赖清单（145 个包，锁版本）
└── venv.txt                            # conda 环境创建指引
```

## 核心模块说明

### 1. OCR 解析（dots_ocr）

对接外部部署的多模态 OCR 模型（DotsOCR，OpenAI 兼容接口，默认 `127.0.0.1:6006`），
把 PDF 按页解析出 **Markdown 文本 + 版面 JSON + 页面图片**，输出到 `output/<文档名>/`。
图片会做超分/压缩（`min_pixels` / `max_pixels`）、版面坐标预处理与 md 结构整理。

### 2. Markdown 目录分割器（MarkdownDirSplitter）

- 按 **Header 1~N 层级** 组织文档结构，先粗切再语义切分（`SemanticChunker`）。
- 语义切分与文档向量化**同模型**（见上文「嵌入模型」），保证切分相似度与检索向量空间一致。
- 从 md 中**提取配图**到 `output/images/`，生成两类 Document：
    - `embedding_type="text"`：纯文本片段（标题已拼接进正文）
    - `embedding_type="image"`：图片片段（`page_content` 为图片路径）

### 3. 多模态向量化（embedding 双后端）

`local` 为默认后端，通过环境变量 `EMBEDDING_BACKEND` 可切换到云端：

| 后端      | 模型                                  | 向量维度   | 说明                   |
|---------|-------------------------------------|--------|----------------------|
| `local` | gme-Qwen2-VL-2B-Instruct（默认）        | 1536   | 本地推理，无限流             |
| `cloud` | DashScope `multimodal-embedding-v1` | 由服务端决定 | 内置 RPM 限流与 429 指数退避重试 |

两种后端都支持纯文本（`text`）/ 纯图片（`image`）/ 图文融合（`text_image`）三种输入模式。
上层 `db_operator` 只依赖 `embedding/common/embedding_selector.py`，不感知具体实现。

### 4. Milvus 向量库（milvus_db）

- **集合 Schema**：主键自增 + `text`（jieba 分词 BM25）+ `category` / `filename` / `filetype` / `image_path` / `title`
    + `sparse`（BM25 函数自动生成）+ `dense`（1536 维，IP 相似度，AUTOINDEX）。
- **写入**：`do_save_to_milvus` 完成 Document → 字典 → 向量化 → 批量写入，
  云端口径支持 429 重试，失败数据自动过滤并记录日志。
- **检索**（`MilvusRetriever`）：

| 方法                     | 用途                           |
|------------------------|------------------------------|
| `dense_search`         | 语义相似检索（1536 维向量）             |
| `sparse_search`        | BM25 关键词全文检索                 |
| `image_search`         | 以图搜图 / 以图搜文（可选 `only_image`） |
| `multimodal_search`    | 文本 + 图片融合为一向量再检索             |
| `hybrid_search`        | 客户端 RRF 融合（两路都命中可加权）         |
| `hybrid_search_native` | 服务端原生 RRF 融合（Milvus 2.4+）    |

### 5. RAG 效果评估（evaluate）

基于 Ragas 对 RAG 链路做量化评估，入口在 `evaluate/`：

| 脚本                        | 指标                                     | 说明                            |
|---------------------------|----------------------------------------|-------------------------------|
| `evaluate_single_turn.py` | 上下文相关性 / 答案相关性 / 上下文精确度                | 上下文精确度支持有/无参考答案两种模式           |
| `evaluate_multi_turn.py`  | 目标达成度 / 主题一致性（f1 / precision / recall） | 目标达成度为 `None` 时标记「评估失败（输出截断）」 |

- **评估 LLM**：GLM（`glm-5.3-flash`，异步 OpenAI 兼容客户端，`max_tokens=4096`）。
- **评估嵌入**：`ModernQwen2Embeddings`（见上文「嵌入模型」）。
- 四个单轮指标统一取自 `ragas.metrics.collections`（新版 API，具备异步 `ascore`）；
  `ragas.metrics` 下的同名指标是旧版实现、只有同步单轮接口，**不要混用**，否则会抛 `AttributeError`。

### 6. 多轮对话 RAG 工作流（graph）

用 **LangGraph** 把「历史上下文检索 → 评估 → 人工审批 → 联网兜底 → 写回向量库」串成一张图，
命令行与 Gradio 两个入口共用同一套节点与图结构。

**状态**（`custom_state.py`，继承 LangGraph 的 `MessagesState`）：
`input_type`（`has_text` / `only_image`）、`context_retrieved`、`image_retrieved`、
`web_search_result`、`evaluate_score`、`human_answer`、`username`、`input_text` / `input_image`。

**节点**

| 节点 | 作用 |
|---|---|
| `process_input` | 判别输入类型（纯文本 / 纯图片 / 图文），写入状态 |
| `first_chatbot` | 绑定 `search_context` 工具的 LLM，决定检索 |
| `search_context` | 自定义工具节点：把状态里的 `username` 显式传给工具并并行执行 |
| `second_chatbot` | 基于检索到的历史对话上下文作答 |
| `third_chatbot` | 纯图片支线：基于 `retriever_node` 的结果作答 |
| `web_search_node` | **确定性**联网检索节点（直接调 `my_search`，不依赖模型自主决定调工具） |
| `fourth_chatbot` | 基于网络搜索结果作答，并给消息打上 `answer_source` 标记 |
| `evaluate_node` | 答案相关性评估（AnswerRelevancy） |
| `human_approval` | 静态中断点，由人工输入 `approve` / `rejected` |

**要点**

- **本地查不到就联网**：`search_context` 未命中（或上下文相关性 < 0.5）时**不进入**
  `retriever_node`，而是直接联网检索；`retriever_node` 只服务纯图片支线。
  原因是它不带任何相关性门槛，会把无价值的命中（如上一轮自己写回的兜底话术）当作"上下文"
  喂给模型，导致回答只能复述兜底话术、被评估判 0 分、进而卡在人工审批。
- **两道分数门槛**：上下文相关性 **0.5**（`tools.py`，判定"本地内容够不够格"）、
  答案相关性 **0.7**（`all_router.py`，判定"要不要转人工审批"）。
- **写回带来源标记**：写库时 `message_type` 取自消息对象上的 `answer_source` ——
  联网所得记 `WebSearch`，历史上下文所得记 `AIMessage`。标记必须挂在**消息**上而非 state 字段：
  state 在同一 thread 内跨轮累积，用字段会把后续轮次一起误标。
- **写库质量闸门**：`save_context.is_worth_saving()` 会拒绝写入兜底话术（命中"没有检索到"等特征）
  或答案相关性低于 `0.3` 的回答。原因：上下文库是「上一轮写、下一轮读」，而兜底话术
  **包含用户的原始提问词**，一旦写回，下一轮同类提问就会命中它 → 上下文评分 0 →
  又生成同样话术 → 又写回去，形成每轮加毒且永不自愈的"冷启动自锁"。

**运行**（必须以**模块方式**从项目根执行；直接 `python graph/workflow.py` 会因项目根不在
`sys.path` 而报 `No module named 'graph'`）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m graph.workflow         # 命令行问答
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m graph.workflow_gradio  # Gradio 界面
```

> 上下文集合需先创建：`python -m graph.graph_db.collections_operator_graph`。
> **该脚本会先 drop 同名集合**（开发期重建用），执行会清空已有数据，日常流程不要跑。

## 环境准备

### 1. 创建 conda 环境

```bash
conda create -n Multimodal_RAG python=3.11 -y
conda activate Multimodal_RAG
pip install -r requirements.txt
```

> 依赖中 `transformers==4.51.3` / `huggingface-hub==0.36.2` / `tokenizers==0.21.4` 必须锁定
> （本地 gme-Qwen2-VL 模型加载要求），它们与 `gradio 6.27.0`、`langchain-experimental 0.4.2`
> 的声明范围存在 2 处硬冲突，**直接一次性安装必然失败**，请按 `venv.txt` 的两步法安装。
> 环境搭建与升级步骤统一见 `venv.txt`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

在 `.env` 中按需填写：

| 变量                             | 说明                                                   |
|--------------------------------|------------------------------------------------------|
| `EMBEDDING_BACKEND`            | `local`（本地 GME，默认）/ `cloud`（DashScope）               |
| `ALIBABA_API_KEY`              | 选用 `cloud` 后端时必填                                     |
| `MILVUS_URI`                   | Milvus 地址，默认 `http://127.0.0.1:19530`                |
| `MILVUS_COLLECTION_NAME`       | 文档集合名，默认 `t_doc_collection`（上下文集合固定为 `t_context_collection`，定义在 graph 代码中） |
| `GLM_API_KEY` / `GLM_BASE_URL` | 评估用 GLM（`glm-5.3-flash`，OpenAI 兼容接口）                  |
| `ZHIPU_API_KEY`                | 智谱 AI 密钥（用于 `init_chat_model_llm.py` 中的 ZhipuAI 客户端） |

> `.env` 已被 `.gitignore` 排除，**严禁提交**，只提交 `.env.example` 模板。

### 3. 启动外部依赖

- **Milvus**：本地 Docker 部署或 Zilliz Cloud，保持与 `MILVUS_URI` 一致。
- **DotsOCR 服务**：自行部署 DotsOCR 并用 vLLM 以 OpenAI 兼容方式启动（默认 `127.0.0.1:6006`）。
- **本地模型**：切换 `EMBEDDING_BACKEND=local` 后，先下载模型：

```bash
python embedding/common/download_model_embedding.py
```

## 运行与使用

全部入口统一在 `Multimodal_RAG` 环境运行，命令均在**项目根目录**执行：

| 入口                                  | 作用            | 说明                                 |
|-------------------------------------|---------------|------------------------------------|
| `milvus_db/collections_operator.py` | 初始化文档集合      | 仅建表，首次使用前执行一次                      |
| `graph/graph_db/collections_operator_graph.py` | 初始化上下文集合 | **会先 drop 同名集合**，慎跑（见下方模块 6）      |
| `splitters/splitters_md.py`         | 分割 → 向量化 → 入库 | 语义切分与文档向量化统一用本地 gme 模型             |
| `main.py`                           | Gradio 交互界面   | 上传 PDF → 解析 → 查看每页 MD → 「存入知识库」    |
| `milvus_db/db_retriever.py`         | 检索测试（含以图搜图）   | 内部直接调用本地 gme 嵌入                    |
| `evaluate/evaluate_*.py`            | RAG 效果评估      | 依赖 `ragas` 包                       |
| `graph/workflow.py`                 | 多轮对话问答（命令行）  | 须用 `python -m graph.workflow`        |
| `graph/workflow_gradio.py`          | 多轮对话问答（界面）   | 须用 `python -m graph.workflow_gradio` |

```bash
conda activate Multimodal_RAG

python milvus_db/collections_operator.py          # ① 初始化集合（仅首次）
python splitters/splitters_md.py                  # ② 构建知识库：分割 → 向量化 → 入库
python main.py                                    # ③ Gradio 界面：上传 PDF → 解析 → 查看 MD → 存入知识库
python milvus_db/db_retriever.py                  # ④ 检索演示
python evaluate/evaluate_single_turn.py           # ⑤ 单轮评估
python evaluate/evaluate_multi_turn.py            # ⑥ 多轮评估

python -m graph.workflow                          # ⑦ 多轮对话问答（命令行，退出输入 退出/exit/quit）
python -m graph.workflow_gradio                   # ⑧ 多轮对话问答（Gradio 界面）
```

- **②** 入口需指定 `md_dir` 与 `images_output_dir`，执行后完成分割、向量化并写入 Milvus。
- **③** 界面流程：上传 PDF → 点击「解析PDF」→ 下拉框查看每页 MD → 点击「存入知识库」，
  与 **②** 等价（同样完成分割、向量化并写入 Milvus）。
- **④** 内置演示：文本语义检索、BM25 关键词检索、RRF 混合检索、以图搜图、图文融合检索。
- **⑦⑧** 多轮对话问答：本地上下文库有该用户的相关记录时直接作答；**查不到会自动联网检索**，
  并把答案写回 `t_context_collection`（联网所得记 `WebSearch`，历史上下文所得记 `AIMessage`），
  因此同一问题问第二次通常可直接命中。两个入口共用同一套节点与图结构，改动需同步。
- 凡会加载本地 gme 模型的入口（**②③④⑤⑥⑦⑧**），在**无外网环境**下建议前置两个环境变量：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py
```

  否则 `local_files_only=True` 仍会对 HuggingFace 发起校验请求并长时间重试（表现为启动明显卡住）。

## 目录数据约定

| 目录               | 是否入库 | 说明                           |
|------------------|------|------------------------------|
| `data/`          | ✔    | 输入数据（PDF、示例图片）               |
| `output/`        | ✔    | OCR 解析结果（每页 md / json / jpg） |
| `output/images/` | ✔    | 分割器提取出的配图（可由代码重新生成）          |
| `logs/`          | ✘    | 运行日志                         |
| `.env`           | ✘    | 密钥等本地配置，见 `.env.example`     |

## Tech Stack

- **解析**：DotsOCR（vLLM）· PyMuPDF
- **分割**：LangChain（MarkdownHeaderTextSplitter / SemanticChunker）
- **向量化**：gme-Qwen2-VL-2B-Instruct（本地）· DashScope multimodal-embedding-v1（云端）
- **向量库**：Milvus 2.4+（BM25 稀疏索引 + AUTOINDEX / IP / RRF）
- **编排**：LangGraph（多轮对话工作流：状态图 + 检查点 + 人工审批中断）
- **联网检索**：智谱 AI `web_search`（search_pro）
- **界面**：Gradio
