import time
from typing import List, Dict

from ragas.llms import llm_factory
from ragas.metrics import ContextRelevance
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextPrecisionWithoutReference

from embedding_demo.custom_embedding import ModernQwen2Embeddings
from milvus_db.collections_operator import client
from milvus_db.db_retriever import MilvusRetriever
from models.init_chat_model_llm import glm_llm_flash, async_glm_llm_flash_client
from utils.env_utils import MILVUS_COLLECTION_NAME
from utils.log_utils import log


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    使用LLM基于检索到的上下文生成文本答案
    Args:
        question: 用户问题
        contexts: 检索到的上下文列表
    Returns:
        str: LLM生成的答案
    """

    # 空上下文兜底：避免提示词退化成"无上下文"，导致 LLM 编造答案
    if not contexts:
        return "没有找到相关的上下文信息，无法回答该问题。"

    # 将检索到的上下文格式化为字符串，便于 LLM 理解。
    # 每个上下文前加上 "上下文X" 标识，方便 LLM 区分不同来源。
    # 注意：contexts 里必须包含 'text' 字段，否则会 KeyError。
    context_str = "\n\n".join([f"上下文 {i + 1}: {context['text']}" for i, context in enumerate(contexts)])

    # 提示词模板（中文版）
    # 约束点：
    #   1. 必须基于上下文回答
    #   2. 不允许添加上下文之外的信息
    # 这是典型的 RAG "grounded generation" 提示词写法。
    prompt = f"""
    你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保你的回答基于提供的上下文，不要添加额外信息。

    用户问题: {question}

    检索到的上下文:
    {context_str}

    请基于以上上下文回答用户问题。
    """

    # 调用 LLM 生成答案
    # glm_llm_flash 应为一个已初始化的 LangChain ChatModel 实例
    # .invoke(prompt) 返回 AIMessage，.content 取纯文本
    response = glm_llm_flash.invoke(prompt)
    return response.content


class RAGEvaluator:
    """
    RAG 的评估类。

    封装了基于 Ragas 的若干评估指标：
      - 上下文相关性 (Context Relevance)
      - 答案相关性 / 响应相关性 (Response Relevancy)
      - 上下文精确度 (Context Precision)，支持有/无参考答案两种模式

    依赖：
      - evaluator_llm:        用于 LLM-as-judge 的评估模型
      - evaluator_embedding:  用于基于嵌入的指标计算
    """

    def __init__(self, evaluator_llm, evaluator_embedding):
        # 评估用的 LLM，通常选用能力较强的模型以保证打分稳定性
        self.evaluator_llm = evaluator_llm
        # 评估用的 embedding 模型，用于 Response Relevancy 等需要向量计算的指标
        self.evaluator_embedding = evaluator_embedding

        # ---- 全部来自 ragas.metrics.collections (新版 API) ----

        # 上下文相关性指标
        self.context_relevance = ContextRelevance(llm=evaluator_llm)

        # 答案相关性指标
        self.answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embedding,
        )

        # 上下文精确度指标分两种：有 / 无参考答案
        self.context_precision_with_ref = ContextPrecision(llm=evaluator_llm)
        self.context_precision_without_ref = ContextPrecisionWithoutReference(llm=evaluator_llm)

    async def evaluate_context(self, question: str, contexts: List[str]) -> float:
        """
        评估上下文的相关性.
        上下文相关性评估: 检索到的上下文（块或段落）是否与用户输入相关。

        Args:
            question: 用户问题
            contexts: 检索到的上下文列表（纯文本列表，不是 Dict）
        Returns:
            float: 上下文相关性评估分数（0.0 ~ 1.0）

        说明：
            底层 LLM 的原始判分等级为 0 / 1 / 2：
                0 → 检索到的上下文与用户查询完全不相关。
                1 → 上下文部分相关。
                2 → 上下文完全相关。
            Ragas 最终会归一化为 0.0 ~ 1.0 的 float 返回。
        """

        result = await self.context_relevance.ascore(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=contexts,  # 检索到的上下文（List[str]）
        )

        # ascore 是异步打分，返回 float
        return result.value

    async def evaluate_answer(self, question: str, contexts: List[Dict], response: str) -> float:
        """
        评估生成的答案（质量）是否与用户输入相关。
        Args:
            question: 用户问题
            contexts: 检索到的上下文列表（Dict 列表，需含 'text' 字段）
            response: LLM生成的答案
        Returns:
            float: 答案相关性（Response Relevancy）评估分数
        """
        log.info(f"[evaluate_answer] 开始，question={question[:50]}")
        t0 = time.time()
        log.info(f"开始评估答案质量, 评估样本为：question={question}, response={response}")
        try:
            result = await self.answer_relevancy.ascore(
                user_input=question,  # 用户输入的问题
                response=response,  # LLM生成的答案
            )
        except Exception as e:
            log.exception(f"[evaluate_answer] 评估失败: {e}")
            raise
        log.info(f"[evaluate_answer] 完成，耗时 {time.time() - t0:.2f}s，分数={result.value}")
        return result.value

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str = None):
        """
        评估RAG的指标
         Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            response: LLM生成的答案
            reference: 可选，参考答案 (用于评估的基准答案，通常为已知的正确答案)
         Returns:
            Dict: 指标值，例如 {"context_precision": 0.85}

         说明：
            - 有 reference → 使用 LLMContextPrecisionWithReference
            - 无 reference → 使用 LLMContextPrecisionWithoutReference
            两者衡量的是"检索到的上下文中，相关内容的排序精度"。
        """

        # 选择评估指标（已预初始化，无需重复构造）
        if reference:
            metric = self.context_precision_with_ref
            result = await metric.ascore(
                user_input=question,  # 用户输入的问题
                retrieved_contexts=[context['text'] for context in contexts],  # 用户输入的问题
                response=response,  # LLM生成的答案
                reference=reference,  # LLM生成的答案
            )
        else:
            metric = self.context_precision_without_ref
            result = await metric.ascore(
                user_input=question,  # 用户输入的问题
                retrieved_contexts=[context['text'] for context in contexts],  # 用户输入的问题
                response=response,  # LLM生成的答案
            )

        print(f"上下文精确度指标的 Score: {result.value}")
        return {"context_precision": result.value}


async def main():
    evaluator_llm = llm_factory("glm-5.3-flash", client=async_glm_llm_flash_client)
    evaluator_embedding = ModernQwen2Embeddings("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")

    # 创建 RAG 评估器
    rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embedding)

    question = "有界流和无界流有什么区别？"

    # 检索上下文 (从Milvus知识库获取)
    m_re = MilvusRetriever(MILVUS_COLLECTION_NAME, client)
    contexts = m_re.retrieve(question)

    generated_answer = generate_answer(question, contexts)
    print(f"生成的答案: {generated_answer}")

    await rag_evaluator.evaluate_metrics(question=question, contexts=contexts, response=generated_answer)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
