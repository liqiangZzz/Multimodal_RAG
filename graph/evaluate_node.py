import asyncio

from langchain_core.messages import AIMessage
from ragas.llms import llm_factory

from embedding.custom_embedding import ModernQwen2Embeddings
from evaluate.evaluate_single_turn import RAGEvaluator
from graph.custom_state import MultimodalRAGState
from models.init_chat_model_llm import async_glm_llm_flash_client
from utils.log_utils import log

# 答案相关性评估的超时上限（秒），与 tools.CONTEXT_EVAL_TIMEOUT 同一动机：
# 评委同样是一次外部 LLM 调用，服务端挂起时会拖到 HTTP 客户端自身的默认超时
# （未显式配置时 OpenAI SDK 约 600s），整轮对话跟着卡死。超时按「低分」处理，
# 交给 human_approval 人工兜底，比闷头等十几分钟好。
# 更换评委模型或供应商后，请依据其响应特征重新评估此值。
ANSWER_EVAL_TIMEOUT = 60

# 评委模型：工作流内的指标打分统一复用这一实例。
# 注意 evaluate/ 下的离线评估脚本会在各自 main() 内单独构造同类实例，
# 所以换模型或供应商时要一并调整，client 的超时 / 重试则在
# models/init_chat_model_llm.py 里按供应商统一配置。
evaluator_llm = llm_factory("glm-5.3-flash", client=async_glm_llm_flash_client)
evaluator_embedding = ModernQwen2Embeddings()

# 创建RAG评估器
rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embedding)


async def evaluate_answer(state: MultimodalRAGState):
    """
        评估大模型的响应和用户输入之间的相关性
    """
    # 注意：本节点只算 AnswerRelevancy（答案与用户问题是否相关），
    # 该指标不消费 contexts；上下文相关性不在这里评，而在 tools.search_context
    # 内部由 rag_evaluator.evaluate_context 把关（门槛见 tools.CONTEXT_SCORE_THRESHOLD）。
    # 因此这里不传 contexts。
    input_text = state.get('input_text')

    last_message = state["messages"][-1]
    answer = last_message.content if isinstance(last_message, AIMessage) else ""

    log.info(f"开始评估答案相关性：{input_text}")
    try:
        score = await asyncio.wait_for(
            rag_evaluator.evaluate_answer(input_text, [], answer),
            timeout=ANSWER_EVAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning(
            f"答案相关性评估超时（>{ANSWER_EVAL_TIMEOUT}s），按低分处理，交由人工审批兜底"
        )
        score = 0.0
    log.info(f"RAG Evaluation Score: {score}")
    return {"evaluate_score": float(score)}
