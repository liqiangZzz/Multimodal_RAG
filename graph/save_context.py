import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple

from pymilvus import MilvusClient

from embedding.custom_embedding import ModernQwen2Embeddings
from graph.graph_db.collections_operator_graph import (
    CONTEXT_COLLECTION_NAME,
    MAX_CONTEXT_TEXT_LENGTH,
    client,
)
from utils.log_utils import log

# 全局线程池用于异步操作
thread_pool = ThreadPoolExecutor(max_workers=5) # 创建一个线程池

embedding = ModernQwen2Embeddings()


# ========== 写库质量闸门 ==========
# 为什么需要这道闸门：
# 上下文库是「上一轮写、下一轮读」。一旦把无价值的回答（尤其是「抱歉，没有检索到…」
# 这类兜底话术）写回去，它会因为**包含用户的原始提问词**而在下一轮同类提问上被命中
# → 上下文评分 0.0 → 又生成同样的话术 → 又写回去。这就是「冷启动自锁」：
# 每轮都在给库加毒，而且永远不会自愈（2026-09-23 实测：库里 2 行全是自己的兜底话术）。
# 闸门放在写入器内部而不是各个调用点，是为了让 graph/workflow.py 与
# graph/workflow_gradio.py 两个入口**同时**受保护，不再出现"只改了一个入口"的问题。

# 兜底话术的特征片段：命中任一即视为无价值内容，拒绝入库。
# 两条链路各有自己的兜底话术，都要覆盖：
#   - 本地检索路径（third_chatbot）：「没有检索到…」系；
#   - 联网兜底路径（fourth_chatbot）：联网也没搜到内容时，模型会按提示词
#     如实回一句「未找到相关网络资料」，my_search 自身则返回「没有搜索到任何内容」。
#     这条路径必须在这里拦住：rejected 之后 evaluate_score 会被作废（那条分数描
#     述的是被否决的答案，见 graph_builder.update_state），此后本特征表就是它
#     唯一的质量闸门，否则这类含用户提问词的空回答会写回库、下一轮被命中，
#     把「冷启动自锁」搬到联网路径上重演。
# 注意只收「整条回答即一句拒绝」的固定措辞，不写宽泛的「未找到相关」，
# 以免误伤正文里正常提到「未找到相关内容」的有效回答。
_FALLBACK_MARKERS = (
    "没有检索到",
    "没有找到相关的历史上下文信息",
    "没有找到相关的上下文信息",
    "未找到相关网络资料",
    "没有找到相关网络资料",
    "没有搜索到任何内容",
    "未能在网络搜索结果中找到",
)

# 答案相关性分数低于该值视为无价值（AnswerRelevancy，取值 0~1）。
# 取 0.3 这种保守值：只拦确定性垃圾，不误伤「人工已 approve 但分数一般」的正常回答。
MIN_EVAL_SCORE_TO_SAVE = 0.3


# ========== 写库去重闸门（幂等）==========
# 为什么需要：
# 「命中检索」这条路径下，second_chatbot 只被允许基于工具返回的历史上下文作答，
# 因此它的输出本质是**库里已有内容的重述**。把这份重述再写回库，等于让同一个答案
# 增殖一份：下次问同样的问题会命中更多同源片段、挤占 top_k，反而稀释真正有用的内容。
# 库只增不改，重复量会随提问次数线性增长 —— 实测该库 16 条里有 7 条是同一个问题的重述。
#
# 判据为什么用向量相似度而不是文本比对：
# 重述会改措辞、改结构，文本比对（甚至归一化后比对）都拦不住，但向量距离很近。
# dense 字段用 IP 且写入的向量已做 L2 归一化，所以 IP 分数可直接当余弦相似度看。
#
# 阈值依据一次实测标定（同一批数据）：
#   - 真实重述（LLM 受「必须基于上下文原文作答」约束的重讲）：0.974 ~ 0.988（三次实测）
#   - 库内已存在的同源重复之间：0.992 ~ 0.999
#   - 不同知识点的记录之间：<= 0.93
# 取 0.95：距真实重述的下沿留约 0.02 余量，与「不同内容」档也保持 0.02 的距离。
# 不取 0.98 的原因：重述相似度会随每次措辞在 0.974~0.988 之间浮动，
# 0.98 正好落在波动区间内，会出现「有时拦住、有时漏拦」的不稳定行为。
#
# 已知局限：这条闸门只对「高度保真的重述」有效。若上下文约束被放宽、
# 模型开始自由改写（实测自由改写相似度可低至 0.93，与「不同知识点」重叠），
# 相似度就再也分不开两者 —— 那种情况下要靠「内容来源」判断，而不是相似度。
#
# 注意：换嵌入模型后向量空间整体改变，此阈值必须重新标定。
DUP_SIM_THRESHOLD = 0.95


# 归一化提问时要抹掉的字符：空白与各类标点。
# 目的只是让「同一问题的不同排版」（多打了空格、句尾问号、全半角混用）落到同一个键上，
# 不做同义词替换 —— 「换个说法问同一件事」由答案向量相似度那条判据兜住。
_QUESTION_NOISE_RE = re.compile(
    r"""[\s?？！!。．，,；;：:、'"“”‘’()（）\[\]【】{}《》<>|\\/`~^*+_=@#$%&—-]+"""
)


def normalize_question(text) -> str:
    """把提问文本归一化成可做等值匹配的幂等键；无有效内容时返回空串。

    写入与判重两端必须调用同一个函数，匹配才自洽。
    """
    if not text:
        return ""
    cleaned = _QUESTION_NOISE_RE.sub("", str(text))
    # 只截断异常超长内容，防止撑爆字段长度；两端同一函数处理，截断后仍能对上
    return cleaned.lower()[:900]


def is_worth_saving(context_text: str, evaluate_score: float | None = None) -> Tuple[bool, str]:
    """判断一条回答是否值得写回历史对话上下文库。

    Returns:
        (是否写入, 跳过原因)。值得写入时原因为空串。
    """
    text = (context_text or "").strip()
    if not text:
        return False, "内容为空"

    for marker in _FALLBACK_MARKERS:
        if marker in text:
            return False, f"命中兜底话术特征「{marker}」"

    if evaluate_score is not None and evaluate_score < MIN_EVAL_SCORE_TO_SAVE:
        return False, f"答案相关性 {evaluate_score} 低于写库阈值 {MIN_EVAL_SCORE_TO_SAVE}"

    return True, ""


class OptimizedMilvusAsyncWriter:
    def __init__(self,
                 client: MilvusClient,
                 collection_name: str = "t_context_collection"):

        self.client = client
        self.collection_name = collection_name

    def _get_dense_vector(self, text: str):
        """异步生成稠密向量"""
        try:

            # 稠密向量生成（假设使用OpenAI或本地模型）
            dense_vector = embedding.embed_text(text)
            return dense_vector

        except Exception as e:
            log.exception(f"向量生成失败: {e}")
            return None

    def _find_duplicate(self, dense_vector, username: str, question: str = ""):
        """检查库中是否已有与本条等价的内容。

        两道判据，任一命中即视为重复：
          1. 问题级：同一用户下已存在归一化后相同的提问 —— 严格意义的幂等，
             「同一个问题问两次」不会再落第二条；
          2. 内容级：答案向量与库内记录的相似度 >= DUP_SIM_THRESHOLD ——
             兜住「换个说法问同一件事」，以及 question 为空的历史数据。
        向量由调用方传入（写入前刚算好的那份），判重不额外编码一次。

        返回 (是否重复, 命中记录 id, 判定说明)；不重复时说明为空串。
        """
        filter_expr = ""
        if username:
            safe_user = username.replace('"', '\\"')
            filter_expr = f'username == "{safe_user}"'

        # ---- 判据 1：问题级等值匹配（交给服务端，客户端零计算）----
        q_key = normalize_question(question)
        if q_key:
            q_filter = (f'{filter_expr} and question == "{q_key}"' if filter_expr
                        else f'question == "{q_key}"')
            try:
                res = self.client.query(
                    collection_name=self.collection_name,
                    filter=q_filter,
                    output_fields=["id"],
                    limit=1,
                )
                if res:
                    return True, res[0]["id"], "问题级命中：库中已有同一提问"
            except Exception as e:
                # 历史数据没有 question 值、或表达式不被支持时，只放弃这一条判据，
                # 不影响下面的内容级判断
                log.warning(f"[Milvus] 问题级去重查询未生效，改用内容级判据: {e}")

        # ---- 判据 2：内容级向量相似度 ----
        try:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field="context_dense",
                limit=1,
                search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                output_fields=["context_text", "timestamp"],
                filter=filter_expr,
                # 与检索侧同样的理由：集合默认的 Bounded 一致性下刚写入的数据查不到，
                # 判据会失效（详见 db_retriever_graph.RetrieverConfig）
                consistency_level="Strong",
            )
        except Exception as e:
            # 去重是优化项而非主链路：判据查询失败时按「不重复」处理，
            # 宁可多写一条，也不要因为查询报错而丢掉本轮的有效回答。
            log.exception(f"[Milvus] 内容级去重检索失败，按不重复处理: {e}")
            return False, None, ""

        if not res or not res[0]:
            return False, None, ""

        top = res[0][0]
        sim = float(top.distance if hasattr(top, "distance") else top.get("distance"))
        hit_id = top.id if hasattr(top, "id") else top.get("id")

        if sim >= DUP_SIM_THRESHOLD:
            return True, hit_id, f"内容级命中：答案相似度 {sim:.4f} >= 阈值 {DUP_SIM_THRESHOLD}"
        return False, hit_id, ""

    def _sync_insert(self, data: Dict[str, Any]):
        """同步插入数据到Milvus"""
        try:
            # 插入数据
            result = self.client.insert(collection_name=self.collection_name, data=data)
            log.info(f"[Milvus] 成功插入 {result['insert_count']} 条记录。IDs 示例: {result['ids'][:5]}")

            # 必须 flush：不 flush 的数据停留在 growing segment，
            # 在集合默认的 Bounded 一致性下**查不到**（row_count 却有值，很有迷惑性），
            # 会导致「上一轮写的历史上下文，下一轮检索不到」。
            # 检索侧已改为 Strong 一致性（见 db_retriever_graph.RetrieverConfig），
            # 这里再 flush 一次，同时保证进程退出后数据已落盘、重启仍在。
            self.client.flush(self.collection_name)
            log.info("[Milvus] flush 完成，数据已可检索")

        except Exception as e:
            log.exception(f"插入数据到Milvus失败: {e}")


    async def async_insert(self, context_text: str, username: str, message_type: str = "AIMessage",
                           evaluate_score: float | None = None, question: str | None = None):
        """异步插入数据。

        写入前会过一道质量闸门（见 is_worth_saving）：无价值的内容一律不入库，
        避免上下文库被自己的兜底话术污染、形成"冷启动自锁"。
        """
        keep, reason = is_worth_saving(context_text, evaluate_score)
        if not keep:
            log.info(f"[Milvus] 跳过写入（{reason}）：{(context_text or '')[:60]!r}")
            return

        # 字段长度保护：context_text 是定长 VARCHAR，超限会被服务端整条拒绝（写入失败）。
        # 这里在入库前按上限截断 —— 截断必须发生在向量计算之前，
        # 保证稠密向量与实际落库的文本一致，内容级去重的相似度才有意义。
        if context_text and len(context_text) > MAX_CONTEXT_TEXT_LENGTH:
            log.warning(
                f"[Milvus] 回答长度 {len(context_text)} 超过字段上限 {MAX_CONTEXT_TEXT_LENGTH}，"
                f"入库前截断（截去 {len(context_text) - MAX_CONTEXT_TEXT_LENGTH} 字符）"
            )
            context_text = context_text[:MAX_CONTEXT_TEXT_LENGTH]

        # 向量只算一次：去重判据与最终写入共用同一个向量
        dense_vector = self._get_dense_vector(context_text)
        if dense_vector is None:
            log.error("[Milvus] 向量生成失败，跳过写入")
            return

        loop = asyncio.get_event_loop()

        # 去重闸门：库中已有同一提问、或高度相似的答案时跳过，避免重复增殖
        is_dup, dup_id, why = await loop.run_in_executor(
            thread_pool, self._find_duplicate, dense_vector, username, question
        )
        if is_dup:
            log.info(f"[Milvus] 跳过写入：{why}（命中 id={dup_id}）：{(context_text or '')[:60]!r}")
            return

        # 准备数据
        data = {
            "context_text": context_text,
            "username": username,
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "message_type": message_type,
            "context_dense": dense_vector,
            # 归一化后的提问作为幂等键；无提问时（例如纯图片输入）留空
            "question": normalize_question(question) or None,
        }

        log.info(f"准备使用线程池异步插入数据: {data}")
        # 使用线程池异步执行插入操作
        await loop.run_in_executor(thread_pool, self._sync_insert, data)


# 全局写入器实例（单例模式）
_milvus_writer_instance = None

def get_milvus_writer() -> OptimizedMilvusAsyncWriter:
    """获取全局Milvus写入器实例（单例）"""
    global _milvus_writer_instance
    if _milvus_writer_instance is None:
        _milvus_writer_instance = OptimizedMilvusAsyncWriter(
            client=client,
            collection_name=CONTEXT_COLLECTION_NAME
        )
    return _milvus_writer_instance