from langchain_core.messages import AIMessage
from ragas.llms import llm_factory

from embedding.custom_embedding import ModernQwen2Embeddings
from evaluate.evaluate_single_turn import RAGEvaluator
from graph.custom_state import MultimodalRAGState
from models.init_chat_model_llm import async_glm_llm_flash_client
from utils.log_utils import log

evaluator_llm = llm_factory("glm-5.3-flash", client=async_glm_llm_flash_client)
evaluator_embedding = ModernQwen2Embeddings()

# 创建RAG评估器
rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embedding)


async def evaluate_answer(state: MultimodalRAGState):
    """
        评估大模型的响应和用户输入之间的相关性
    """
    # 注意：本节点只算 AnswerRelevancy（答案与用户问题是否相关），
    # 该指标不消费 contexts；上下文相关性由 tools.search_context 内部的
    # rag_evaluator.evaluate_context（阈值 0.5）负责把关。
    # 因此这里不再拼装 contexts（原先从 ToolMessage 回填那段是死代码）。
    input_text = state.get('input_text')

    last_message = state["messages"][-1]
    answer = last_message.content if isinstance(last_message, AIMessage) else ""

    score = await rag_evaluator.evaluate_answer(input_text, [], answer)
    log.info(f"RAG Evaluation Score: {score}")
    return {"evaluate_score": float(score)}
