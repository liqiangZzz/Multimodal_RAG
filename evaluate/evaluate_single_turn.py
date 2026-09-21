from typing import List, Dict

from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics._context_precision import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference

from embedding_demo.custom_embedding import CustomQwen3Embeddings
from milvus_db.collections_operator import client
from milvus_db.db_retriever import MilvusRetriever
from models.init_chat_model_llm import glm_llm_flash, glm_llm_flash_client
from utils.env_utils import MILVUS_COLLECTION_NAME


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    使用LLM基于检索到的上下文生成文本答案
    Args:
        question: 用户问题
        contexts: 检索到的上下文列表
    Returns:
        str: LLM生成的答案
    """

    # 将检索到的上下文格式化为字符串，便于LLM理解
    # 每个上下文前加上"上下文X"标识，方便LLM区分
    context_str = "\n\n".join([f"上下文 {i + 1}: {context['text']}" for i, context in enumerate(contexts)])

    # 提示词模版 (已翻译成中文)
    prompt = f"""
    你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保你的回答基于提供的上下文，不要添加额外信息。

    用户问题: {question}

    检索到的上下文:
    {context_str}

    请基于以上上下文回答用户问题。
    """

    # 调用 LLM 生成答案

    response = glm_llm_flash.invoke(prompt)
    return response.content


class RAGEvaluator:
    """
    RAG的评估类
    """

    def __init__(self, evaluator_llm, evaluator_embedding):
        self.evaluator_llm = evaluator_llm
        self.evaluator_embedding = evaluator_embedding

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str = None):
        """
        评估RAG的指标
         Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            response: LLM生成的答案
            reference: 可选，参考答案 (用于评估的基准答案，通常为已知的正确答案)
         Returns:
            Dict: 指标值
        """

        # 1. 创建评估样本（SingleTurnSample）
        sample = SingleTurnSample(
            user_input=question,  # 用户输入的问题
            retrieved_contexts=[context['text'] for context in contexts],  # 检索到的上下文
            response=response,  # LLM生成的答案
            reference=reference  # 参考答案，用于评估的基准答案，通常为已知的正确答案
        )

        # 2. 初始化评估指标
        if reference:
            # 如果有参考答案，则初始化指标为LLMContextPrecisionWithReference
            context_precision = LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        else:
            # 如果没有参考答案，则初始化指标为LLMContextPrecisionWithoutReference
            context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)

        # 3. 执行评估指标得到结果
        context_precision_score = await context_precision.single_turn_ascore(sample)
        print(f"上下文精确度指标的 Score: {context_precision_score}")


async def main():
    evaluator_llm = llm_factory("glm-5.3-flash", client=glm_llm_flash_client)
    evaluator_embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")

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
