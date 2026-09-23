from langgraph.constants import END

from graph.custom_state import MultimodalRAGState


def route_only_image(state: MultimodalRAGState):
    """动态路由函数，如果用户仅仅输入图片，则进入LLM节点，否则进入历史对话上下文检索节点"""

    if state.get("input_type") == "only_image":
        return "retriever_node"

    return "first_chatbot"


def route_llm_or_retriever(state: MultimodalRAGState):
    """
    动态路由函数：
    - 命中历史上下文 -> second_chatbot（基于工具返回的上下文作答）
    - 未命中（工具返回空 / 低于 0.5 分门槛被清空）-> retriever_node（放宽条件再检索一遍，
      结果写入 state['context_retrieved']，交给 third_chatbot 作答）

    联网不是在这一步触发的：本地答不好会先经过 evaluate_node 评估，
    分数低于 0.7 走人工审批，用户 rejected 之后才由 fourth_chatbot 联网兜底重新作答。
    """
    if messages := state.get("messages", []):
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")

    if not tool_message.content or tool_message.content == "没有找到相关的历史上下文信息。":
        return "retriever_node"
    return 'second_chatbot'


def route_evaluate_node(state: MultimodalRAGState):
    """动态路由函数，如果用户仅仅输入图片，则不进行评估，其他情况下进入评估节点"""
    if state.get('input_type') == 'only_image':
        return END
    return 'evaluate_node'


def route_human_node(state: MultimodalRAGState):
    """动态路由函数，如果评估后的分值低于0.7，则进入人工介入节点 """

    if state.get('evaluate_score') is not None and state.get('evaluate_score') >= 0.7:
        return END
    return 'human_approval'

def route_human_approval_node(state: MultimodalRAGState):
    """
    动态路由函数，如果用户输入的是：approve 则结束，否则进入网络搜索
    """

    if state.get('human_answer') == 'approve':
        return END
    return 'fourth_chatbot'
