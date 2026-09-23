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
    - 命中历史上下文 -> second_chatbot（基于上下文作答）
    - 未命中（本地上下文库没有可用内容）-> web_search_node（直接联网检索，结果随后写回向量库）

    说明：未命中时**不再**进入 retriever_node。
    retriever_node 不做任何相关性门槛，会把无价值的命中（例如上一轮自己写回的兜底话术）
    当作"上下文"喂给模型，导致回答只能复述兜底话术，进而被评估判 0 分、卡在人工审批。
    联网兜底才是"本地没有"时的正确出口。retriever_node 仍保留给 only_image 路径使用。
    """
    if messages := state.get("messages", []):
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")

    if not tool_message.content or tool_message.content == "没有找到相关的历史上下文信息。":
        return "web_search_node"
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
