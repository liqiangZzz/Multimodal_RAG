import asyncio
import time
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from embedding.gme_qwen2_vl_2b_embedding import call_local_model
from graph.evaluate_node import rag_evaluator
from graph.graph_db.collections_operator_graph import CONTEXT_COLLECTION_NAME, client
from graph.graph_db.db_retriever_graph import MilvusRetriever, RetrieverConfig
from models.init_chat_model_llm import zhipuai_client
from utils.log_utils import log

# 上下文相关性评估的超时上限（秒）。
# 这道评估是「附加判定」而不是主链路：它只决定要不要丢掉本轮检索结果。
# 评委是一次外部 LLM 调用，服务端偶发不响应时，未显式配置超时的 HTTP 客户端
# 会一直等到自身默认超时才返回（OpenAI SDK 未配置时约 600s）。而 search_context
# 是 first_chatbot 的必经节点，一旦挂住，整轮对话会跟着卡十几分钟，
# 期间 Gradio 界面没有任何反馈（2026-09-23 实测过一次约 600s 的挂起）。
# 所以给一个显式上限：超时视为「无法判定」，放行本轮检索结果 —— 宁可少一道闸门也不能卡死。
# 更换评委模型或供应商后，请依据其响应特征重新评估此上限。
CONTEXT_EVAL_TIMEOUT = 30

# 上下文相关性评分门槛：低于该值视为「检索结果与本轮问题无关」，直接返回空。
# 注意 route_llm_or_retriever 依赖「返回空」来判定未命中，调整此值会改变路由行为。
CONTEXT_SCORE_THRESHOLD = 0.5


@tool("search_context", parse_docstring=True)
async def search_context(
        query: Optional[str] = None,
        username: Optional[str] = None
) -> str:
    """
    根据用户的输入，检索与查询相关的历史上下文信息，然后给出正确的回答。

    Args:
        query: (可选) 用户输入的查询
        username: (可选) 用户名

    Returns:
        str: 检索到的历史上下文信息
    """
    if not query:
        return "没有找到相关的历史上下文信息。"

    try:
        # 1. 文本 -> 密集向量
        # 构建文本输入数据
        input_data = [{"text": query}]

        # 调用API获取嵌入向量
        ok, embedding, status, retry_after = call_local_model(input_data)

        if not ok or embedding is None:
            log.error(f"Embedding 调用失败: status={status}, retry_after={retry_after}")
            return "没有找到相关的历史上下文信息。"

        # 2. 过滤表达式
        filter_expr = ''
        if username:
            safe_user = username.replace('"', '\\"')
            filter_expr = f'username == "{safe_user}"'

        # 3. 初始化检索器，可以不传默认使用默认配置。
        config = RetrieverConfig(
            top_k=5,
            dense_nprobe=10,
            sparse_drop_ratio=0.2,
            rrf_k=60,
            both_hit_bonus=1.2
        )

        retriever = MilvusRetriever(
            collection_name=CONTEXT_COLLECTION_NAME,
            milvus_client=client,
            config=config,
        )

        # 4. 混合检索
        hits = retriever.hybrid_search(
            query_text=query,
            query_embedding=embedding,
            limit=5,
            rrf_k=60,
            filter_expr=filter_expr,
        )

        log.info(f'上下文检索结果：{hits}')

        # 5. 抽取 context_text，并过滤空值（字段名以实际返回结构为准）
        context_pieces = [
            (hit.get('entity') or hit).get('context_text')
            for hit in hits
        ]
        context_pieces = [c for c in context_pieces if c]

        if not context_pieces:
            return "没有找到相关的历史上下文信息。"

        # 6. 调用上下文相关性指标评估（外部 LLM 打分，必须有超时，见 CONTEXT_EVAL_TIMEOUT）
        t_eval = time.time()
        try:
            score = await asyncio.wait_for(
                rag_evaluator.evaluate_context(query, context_pieces),
                timeout=CONTEXT_EVAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                f"上下文相关性评估超时（>{CONTEXT_EVAL_TIMEOUT}s），"
                f"本轮跳过 {CONTEXT_SCORE_THRESHOLD} 分门槛，直接使用检索结果"
            )
            score = 1.0
        log.info(f"上下文检索后，评估分数为: {score}（评估耗时 {time.time() - t_eval:.2f}s）")

        if score < CONTEXT_SCORE_THRESHOLD:  # 低于门槛，视为与本轮问题无关
            context_pieces = []
        return "\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息。"

    except Exception as e:
        log.exception(f"search_context 执行失败: {e}")
        return "没有找到相关的历史上下文信息。"


class SearchInput(BaseModel):
    query: str = Field(description='需要搜索的内容或者关键词')


@tool("my_search", args_schema=SearchInput, description="根据用户输入的查询，进行网络搜索")
def my_search(query: str) -> str:
    """搜索互联网上所有的公开内容"""
    try:
        response = zhipuai_client.web_search.web_search(
            search_engine="search_pro",
            search_query=query
        )

        # SDK 的返回可能是对象或 dict，search_result 可能是单个对象/字典，也可能是列表
        results = getattr(response, "search_result", None)
        if results is None and isinstance(response, dict):
            results = response.get("search_result")
        if not results:
            return '没有搜索到任何内容！'

        def _content(item):
            if isinstance(item, dict):
                return item.get("content")
            return getattr(item, "content", None)

        if isinstance(results, (list, tuple)):
            text_parts = [c for c in (_content(d) for d in results) if c]
            return "\n\n".join(text_parts) if text_parts else '没有搜索到任何内容！'

        content = _content(results)
        return content if content else '没有搜索到任何内容！'
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'