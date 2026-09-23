"""
多轮对话 RAG 评估脚本（适配 Ragas 0.4.3 collections API）。

本脚本使用 Ragas 0.4.3 的现代指标系统，核心特征如下：
- 指标从 ragas.metrics.collections 导入；
- 对话历史以 List[HumanMessage | AIMessage] 的形式直接传入 ascore()；
- 无需构造 MultiTurnSample 包装对象；
- 评分方法为 await metric.ascore(**kwargs)，返回 MetricResult 对象；
- 通过 result.value 获取分数，result.reason 获取 LLM 给出的评分理由。
"""
from typing import List, Dict, Literal, get_args

from ragas.llms import llm_factory
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AgentGoalAccuracyWithoutReference,
    TopicAdherence,
)

from embedding.custom_embedding import ModernQwen2Embeddings
from milvus_db.collections_operator import client
from milvus_db.db_retriever import MilvusRetriever
from models.init_chat_model_llm import glm_llm_flash, async_glm_llm_flash_client
from utils.env_utils import MILVUS_COLLECTION_NAME

# 定义 TopicAdherence 支持的 mode 类型别名
TopicMode = Literal["precision", "recall", "f1"]


# ---------------------------------------------------------------------------
# 1. RAG 生成（支持多轮历史）
# ---------------------------------------------------------------------------
def generate_answer(
        question: str,
        contexts: List[Dict],
        history: List[Dict] = None,
) -> str:
    """
    使用 LLM 基于检索到的上下文和对话历史生成答案。

    Args:
        question: 用户当前问题
        contexts: 检索到的上下文列表，每项为 {"text": "..."}
        history:  可选，之前的对话历史，格式
                  [{"role": "human"/"ai", "content": "..."}, ...]
    Returns:
        str: LLM 生成的答案
    """
    context_str = "\n\n".join(
        [f"上下文 {i + 1}: {ctx['text']}" for i, ctx in enumerate(contexts)]
    )

    history_str = ""
    if history:
        lines = []
        for h in history:
            speaker = "用户" if h["role"] == "human" else "助手"
            lines.append(f"{speaker}: {h['content']}")
        history_str = "\n之前的对话历史:\n" + "\n".join(lines) + "\n"

    prompt = f"""
    你是一个AI助手，需要根据提供的上下文回答用户的问题。
    请确保你的回答基于提供的上下文，不要添加额外信息。
    {history_str}
    用户问题: {question}

    检索到的上下文:
    {context_str}

    请基于以上上下文回答用户问题。
    """

    response = glm_llm_flash.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# 2. 多轮 RAG 评估器
# ---------------------------------------------------------------------------
class MultiTurnRAGEvaluator:
    """多轮对话 RAG 评估器"""

    def __init__(self, evaluator_llm, evaluator_embedding):
        self.evaluator_llm = evaluator_llm
        self.evaluator_embedding = evaluator_embedding

    @staticmethod
    def _to_ragas_messages(conversation: List[Dict]) -> List:
        """将 [{"role": "human"/"ai", "content": "..."}] 转为 Ragas 消息列表。"""
        messages = []
        for turn in conversation:
            role = turn["role"].lower()
            if role == "human":
                messages.append(HumanMessage(content=turn["content"]))
            elif role == "ai":
                messages.append(AIMessage(content=turn["content"]))
            else:
                raise ValueError(f"未知的角色: {turn['role']}（仅支持 'human' / 'ai'）")
        return messages

    # ---------- 目标达成度评估 ----------
    async def evaluate_goal_accuracy(
            self,
            conversation: List[Dict],
            reference: str = None
    ) -> dict:
        """
            评估多轮对话的最终目标是否达成。
        Args:
            conversation: 多轮对话历史，每个轮次包含用户输入和模型回复。
            reference: 可选，参考答案。
                       - 有 reference：使用 AgentGoalAccuracyWithReference
                       - 无 reference：使用 AgentGoalAccuracyWithoutReference
        Returns:
            float: 目标达成度指标值
        """

        # 1. 将对话转换为消息列表
        messages = self._to_ragas_messages(conversation)

        # 2. 选择并初始化指标
        if reference:
            metric = AgentGoalAccuracyWithReference(llm=self.evaluator_llm)
            # 3. 使用 ascore()，直接传入消息和参考目标
            result = await metric.ascore(user_input=messages, reference=reference)
        else:
            metric = AgentGoalAccuracyWithoutReference(llm=self.evaluator_llm)
            # 3. 使用 ascore()，直接传入消息列表
            result = await metric.ascore(user_input=messages)

        # 4. 从 MetricResult 中提取分数
        print(f"[目标达成度] Score: {result.value}")
        if result.reason:
            print(f"[目标达成度] Reason: {result.reason}")

        return {"score": result.value, "reason": result.reason}

    # ---------- 主题一致性评估 ----------
    async def evaluate_topic_adherence(self, conversation: List[Dict], reference_topics: List[str],
                                       mode: TopicMode = "f1", ) -> dict:
        """
        评估主题一致性
        Args:
            conversation: 多轮对话历史，每个轮次包含用户输入和模型回复。
            reference_topics: 参考的主题列表。
            mode: 主题一致性指标的计算模式。
                       - "f1": 计算 F1 分数（默认）
                       - "accuracy": 计算准确率
                       - "precision": 计算精确率
                       - "recall": 计算召回率
                       - "micro": 计算微平均 F1 分数
                       - "macro": 计算宏平均 F1 分数
        Returns:
            dict: 主题一致性指标值和理由的字典。
                       - "score": 主题一致性指标值
                       - "reason": LLM 给出的评分理由
        """

        messages = self._to_ragas_messages(conversation)

        metric = TopicAdherence(llm=self.evaluator_llm, mode=mode)
        result = await metric.ascore(user_input=messages, reference_topics=reference_topics)

        print(f"[主题一致性-{mode}] Score: {result.value}")
        if result.reason:
            print(f"[主题一致性-{mode}] Reason: {result.reason}")

        return {"score": result.value, "reason": result.reason}

    # ---------- 多轮评估 ----------
    async def evaluate_multi_turn(self, conversation: List[Dict], reference: str = None,
                                  reference_topics: List[str] = None) -> Dict[str, float]:
        """
        对一段多轮对话执行所有可用指标，返回指标字典。
        Args:
            conversation: 多轮对话历史，每个轮次包含用户输入和模型回复。
            reference: 可选，参考答案。
                       - 有 reference：使用 AgentGoalAccuracyWithReference
                       - 无 reference：使用 AgentGoalAccuracyWithoutReference
            reference_topics: 参选，参考主题列表(用于 TopicAdherence)
        Returns:
            dict: 所有指标的字典。
                       - "score": 指标值
                       - "reason": LLM 给出的评分理由
        """

        results = {}

        # 目标达成度
        goal_result = await self.evaluate_goal_accuracy(
            conversation, reference=reference
        )
        # 目标达成度指标值
        if goal_result["score"] is not None:
            results["agent_goal_accuracy"] = goal_result["score"]
        else:
            results["agent_goal_accuracy"] = "评估失败（输出截断）"

        # 主题一致性（分别以 f1、precision、recall 三种模式评估）
        if reference_topics:
            for mode in get_args(TopicMode):
                # 主题一致性
                topic_result = await self.evaluate_topic_adherence(
                    conversation, reference_topics, mode=mode
                )
                # 主题一致性指标值
                results[f"topic_adherence_{mode}"] = topic_result["score"]

        return results


# ---------------------------------------------------------------------------
# 3. 构造多轮 RAG 交互序列
# ---------------------------------------------------------------------------
def build_multi_turn_rag_conversation(
        questions: List[str],
        retriever: MilvusRetriever,
) -> List[Dict]:
    """
    依次对多轮问题执行 RAG（检索 + 生成），返回完整对话历史。
    Args:
        questions: 按顺序排列的多轮用户问题
        retriever: MilvusRetriever 实例

    Returns:
        List[Dict]: [{"role": "human"/"ai", "content": "..."}, ...] 多轮 RAG 交互序列
    """

    conversation: List[Dict] = []
    for question in questions:
        # 用户输入
        conversation.append({"role": "human", "content": question})
        # 检索上下文
        contexts = retriever.retrieve(question)
        # 生成回复
        answer = generate_answer(question, contexts, history=conversation[:-1])
        # 模型回复
        conversation.append({"role": "ai", "content": answer})

    return conversation


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
async def main():
    # ---------- 初始化评估 LLM ----------
    # 注意：async_glm_llm_flash_client 必须是 AsyncOpenAI 兼容的异步客户端
    evaluator_llm = llm_factory("glm-5.3-flash", client=async_glm_llm_flash_client,max_tokens=4096)
    evaluator_embedding = ModernQwen2Embeddings()

    multi_turn_evaluator = MultiTurnRAGEvaluator(
        evaluator_llm, evaluator_embedding
    )

    # ---------- 构造多轮问题 ----------
    questions = [
        "有界流和无界流有什么区别？",
        "那它们在状态管理上有什么不同？",
        "无界流场景下如何保证 exactly-once 语义？",
    ]

    # ---------- 执行多轮 RAG ----------
    retriever = MilvusRetriever(MILVUS_COLLECTION_NAME, milvus_client=client)
    conversation = build_multi_turn_rag_conversation(questions, retriever)
    print("=" * 60)
    print("对话历史:")
    for turn in conversation:
        speaker = "用户" if turn["role"] == "human" else "助手"
        print(f"  [{speaker}] {turn['content']}")
    print("=" * 60)

    # ---------- 评估参数 ----------
    reference = (
        "用户希望了解有界流与无界流的区别、状态管理差异，"
        "以及无界流下 exactly-once 的实现方式。"
    )
    reference_topics = ["流处理", "状态管理", "exactly-once"]

    # ---------- 执行多轮评估 ----------
    results = await multi_turn_evaluator.evaluate_multi_turn(
        conversation,
        reference=reference,
        reference_topics=reference_topics,
    )

    print("\n" + "=" * 60)
    print("多轮评估结果:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
