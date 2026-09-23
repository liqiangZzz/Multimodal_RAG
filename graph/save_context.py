import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple

from pymilvus import MilvusClient

from embedding.custom_embedding import ModernQwen2Embeddings
from graph.graph_db.collections_operator_graph import CONTEXT_COLLECTION_NAME, client
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
_FALLBACK_MARKERS = (
    "没有检索到",
    "没有找到相关的历史上下文信息",
    "没有找到相关的上下文信息",
)

# 答案相关性分数低于该值视为无价值（AnswerRelevancy，取值 0~1）。
# 取 0.3 这种保守值：只拦确定性垃圾，不误伤「人工已 approve 但分数一般」的正常回答。
MIN_EVAL_SCORE_TO_SAVE = 0.3


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
                           evaluate_score: float | None = None):
        """异步插入数据。

        写入前会过一道质量闸门（见 is_worth_saving）：无价值的内容一律不入库，
        避免上下文库被自己的兜底话术污染、形成"冷启动自锁"。
        """
        keep, reason = is_worth_saving(context_text, evaluate_score)
        if not keep:
            log.info(f"[Milvus] 跳过写入（{reason}）：{(context_text or '')[:60]!r}")
            return

        # 准备数据
        data = {
            "context_text": context_text,
            "username": username,
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "message_type": message_type,
            "context_dense": self._get_dense_vector(context_text)
        }

        log.info(f"准备使用线程池异步插入数据: {data}")
        # 使用线程池异步执行插入操作
        loop = asyncio.get_event_loop()
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