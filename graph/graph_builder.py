"""多轮对话 RAG 工作流的**唯一**图定义。

workflow.py（命令行）与 workflow_gradio.py（Gradio 界面）都从这里导入节点、路由与编译好的图，
各自只保留输入 / 输出外壳。此前两个文件各自复制了一份节点函数与图结构，改一处漏一处，
出现过「命令行改好了、界面还是旧逻辑」的问题；现在图只有一份，两个入口不可能再走偏。

图结构（与设计图一致）：

    START          -> process_input
    process_input  -> first_chatbot（含文本）| retriever_node（仅图片）
    first_chatbot  -> search_context（tools）| END
    search_context -> retriever_node（未命中）| second_chatbot（命中，终态）
    retriever_node -> third_chatbot
    third_chatbot  -> evaluate_node | END
    evaluate_node  -> human_approval | END
    human_approval -> fourth_chatbot（rejected）| END（approve）
    fourth_chatbot -> web_search_node（tools）| END
    web_search_node-> fourth_chatbot
"""

import os
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore

from graph.all_router import (
    route_evaluate_node,
    route_human_approval_node,
    route_human_node,
    route_llm_or_retriever,
    route_only_image,
)
from graph.common import InvalidInputError
from graph.custom_state import MultimodalRAGState
from graph.evaluate_node import evaluate_answer
from graph.save_context import get_milvus_writer
from graph.search_node import SearchContextToolNode, retriever_node
from graph.tools import my_search, search_context
from models.init_chat_model_llm import glm_llm_flash
from utils.log_utils import log

# 历史对话上下文检索工具
context_tools = [search_context]
# 网络搜索工具（联网兜底用）
web_tools = [my_search]


# ============== 工作流节点函数 ==============
def process_input(state: MultimodalRAGState, config: RunnableConfig):
    """处理用户输入，判断本轮是「含文本」还是「仅图片」。"""
    # 从配置对象中获取用户名
    username = config["configurable"].get("username", "ZS")
    # 从状态对象中获取最后一个消息，HumanMessage 类型，用户输入的消息
    last_message = state["messages"][-1]

    log.info(f"用户 {username} 输入：{last_message.content}")

    input_type = "has_text"
    text_content = None
    image_url = None

    # 检查输入的类型
    if isinstance(last_message, HumanMessage):
        # 多模态的消息，包含文本和图像
        if isinstance(last_message.content, list):
            content = last_message.content
            for item in content:
                # 提取文本内容
                if item.get("type") == "text":
                    text_content = item.get("text", None)
                # 提取图像 URL（base64 data URI 或在线 url）
                elif item.get("type") == "image_url":
                    url = item.get("image_url", "").get("url")
                    if url:
                        image_url = url
    else:
        raise InvalidInputError(f"用户输入的消息错误！原始输入：{last_message}")

    # 仅包含图像
    if not text_content and image_url:
        input_type = "only_image"

    return {
        "input_type": input_type,
        "username": username,
        "input_text": text_content,
        "input_image": image_url,
    }


def first_chatbot(state: MultimodalRAGState):
    """第一次生成回复或者决策（基于当前会话生成回复），负责触发历史上下文检索工具。"""
    llm_with_tools = glm_llm_flash.bind_tools(context_tools)

    system_message = SystemMessage(content="""你是一名专精于 Apache Flink 的 AI 助手，可以调用工具调取与该用户的历史对话记录。

    # 工具说明（务必准确理解）：
    `search_context` 检索的是**该用户的历史对话上下文**（对话记忆），不是 Flink 文档知识库；
    返回内容为过去对话中的相关片段，可能为空。

    # 核心指令（必须严格遵守）：
    1.**首要规则**：当用户提问涉及 Apache Flink 的任务技术概念、配置、代码或实践时，你**必须且只能**调用 `search_context` 工具来获取信息。
    2.**禁止行为**：你**严禁**凭借自身内部知识直接回答任何关于 Flink 的技术问题。你的回答必须完全基于工具返回的历史对话内容。
    3.**表述要求**：工具返回的是历史对话记录，不要把它描述成“文档”“资料”或“知识库文档”。
    4.**兜底策略**：如果工具返回了相关信息，请基于这些信息组织答案。如果工具明确返回“没有找到相关的历史上下文信息。”，你应统一回复：“关于这个问题，我当前的历史对话记录中没有找到确切的资料。”

    # 回答流程（不可更改）：
    用户提问 -> 调用 `search_context` 工具 -> 基于工具返回结果生成答案。
    """)

    return {"messages": [llm_with_tools.invoke([*state["messages"], system_message])]}


def second_chatbot(state: MultimodalRAGState):
    """第二次生成回复：基于检索到的历史上下文作答（检索结果在 ToolMessage 里）。"""
    return {"messages": [glm_llm_flash.invoke(state["messages"])]}


def third_chatbot(state: MultimodalRAGState):
    """第三次生成回复：基于 retriever_node 检索到的历史对话上下文作答。"""
    # 从向量数据库中检索到的文本内容
    context_retrieved = state.get("context_retrieved")
    # 从向量数据库中检索到的图像 URL 路径
    image_retrieved = state.get("image_retrieved")

    # 注意：retriever_node 检索的是 t_context_collection（历史对话上下文库），
    # 返回字段为 context_text / username / timestamp / message_type，
    # 不存在 text / filename（那是文档库 t_doc_collection 的字段），切勿混用。
    count = 0
    context_pieces = []
    for hit in (context_retrieved or []):
        count += 1
        content = hit.get("context_text")
        source = hit.get("username") or "未知来源"
        context_pieces.append(f"检索后的内容{count}：\n {content} \n 资料来源：{source}")
    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息。"

    input_text = state.get("input_text")
    input_image = state.get("input_image")

    system_prompt = f"""
        请根据用户输入和以下检索到的「历史对话上下文」生成响应。
        注意：这些内容是从该用户的历史对话记录中检索出的片段，不是文档或知识库资料。
        如果上下文内容中没有相关答案，请直接说明，不要自己直接输出答案。
        要求：
        1. 响应必须使用Markdown格式
        2. 在响应文字下方显示所有相关图片，图片的路径列表为{image_retrieved}，使用Markdown图片语法：
        3. 在相关图片下面的最后一行显示上下文引用来源
        4. 如果用户还输入了图片，请也结合上下文内容，生成文本响应内容。
        5. 如果用户还输入了文本，请结合上下文内容，生成文本响应内容。
        6. 不要使用“知识库”“文档”“上传资料”这类字眼，统一表述为“历史对话记录”。

        历史对话上下文：
        {context}
        """

    # 构建用户消息内容
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_content),
        ]
    )

    chain = prompt | glm_llm_flash

    return {"messages": [chain.invoke({"context": context})]}


def human_approval(state: MultimodalRAGState):
    """人工审批节点（静态中断点，interrupt_before=['human_approval']）。"""
    log.info("已经进入了人工审批节点")
    log.info(f"当前的状态中的人工审批信息：{state['human_answer']}")


def fourth_chatbot(state: MultimodalRAGState):
    """第四次模型调用：绑定网络搜索工具，被拒后联网重新作答。

    本节点与 web_search_node 构成回路：fourth_chatbot -> web_search_node -> fourth_chatbot。
    第二次进入时模型必须能看到上一轮的 AIMessage.tool_calls 与 web_search_node 写入的
    ToolMessage，才能把搜索结果用于作答；因此这里把完整历史一起送给模型，
    只发一条新消息会让它看不到结果、反复调工具直到撞上递归上限。
    """
    llm_with_tools = glm_llm_flash.bind_tools(web_tools)

    system_message = SystemMessage(content=(
        "你是一个智能体助手。请**先调用 `my_search` 互联网搜索工具**获取资料，"
        "再基于搜索结果生成回复。\n"
        "要求：\n"
        "1. 使用 Markdown 格式作答，并尽量标注信息来源。\n"
        "2. 若搜索结果中没有相关内容，请如实说明「未找到相关网络资料」，不要编造。"
    ))

    history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    return {"messages": [llm_with_tools.invoke([system_message, *history])]}


# ============== 图构建 ==============
checkpointer = InMemorySaver()
store = InMemoryStore()


def build_graph():
    """按设计图构建并编译工作流。"""
    builder = StateGraph(MultimodalRAGState)

    # 添加节点
    builder.add_node("process_input", process_input)
    builder.add_node("first_chatbot", first_chatbot)
    builder.add_node("search_context", SearchContextToolNode(tools=context_tools))
    builder.add_node("retriever_node", retriever_node)
    builder.add_node("second_chatbot", second_chatbot)
    builder.add_node("third_chatbot", third_chatbot)
    builder.add_node("evaluate_node", evaluate_answer)
    builder.add_node("human_approval", human_approval)
    builder.add_node("fourth_chatbot", fourth_chatbot)
    builder.add_node("web_search_node", ToolNode(tools=web_tools))

    # 添加边
    builder.add_edge(START, "process_input")
    builder.add_conditional_edges("process_input", route_only_image,
                                  {"retriever_node": "retriever_node", "first_chatbot": "first_chatbot"})

    builder.add_conditional_edges("first_chatbot", tools_condition,
                                  {"tools": "search_context", END: END})

    builder.add_conditional_edges("search_context", route_llm_or_retriever,
                                  {"retriever_node": "retriever_node", "second_chatbot": "second_chatbot"})

    builder.add_edge("retriever_node", "third_chatbot")

    builder.add_conditional_edges("third_chatbot", route_evaluate_node,
                                  {"evaluate_node": "evaluate_node", END: END})
    builder.add_conditional_edges("evaluate_node", route_human_node,
                                  {"human_approval": "human_approval", END: END})
    builder.add_conditional_edges("human_approval", route_human_approval_node,
                                  {"fourth_chatbot": "fourth_chatbot", END: END})

    # 被拒后：模型自主调用联网工具 -> web_search_node 执行 -> 回到 fourth_chatbot 作答
    builder.add_conditional_edges("fourth_chatbot", tools_condition,
                                  {"tools": "web_search_node", END: END})
    builder.add_edge("web_search_node", "fourth_chatbot")

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        # 静态人工介入：恢复工作流时从中断点继续
        interrupt_before=["human_approval"],
    )


graph = build_graph()


def export_graph_png(g, filename: str = "graph_rag.png") -> str:
    """导出流程图，按本模块所在目录写出，避免随工作目录漂移。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, "wb") as f:
        f.write(g.get_graph().draw_mermaid_png())
    return path


# 导出流程图（两个入口导入本模块时即刷新，保证图与代码同步）
export_graph_png(graph)


def new_run_config(username: str = "ZS") -> dict:
    """新建一次会话的运行配置（新 thread_id 避免检查点互相污染）。"""
    return {
        "configurable": {
            # 键名必须与 process_input 读取的键一致（config["configurable"].get("username")）
            "username": username,
            "thread_id": str(uuid.uuid4()),
        }
    }


def update_state(user_answer: str, config: dict) -> None:
    """在工作流外面的普通函数中，让人工介入。

    rejected 表示「否决这条答案，改由联网兜底重新生成」，因此要同时**作废
    evaluate_score**：那个分数描述的是刚被否决的答案，对即将联网生成的新答案没有意义。
    不清除的话收尾写库会拿旧分数去判定新答案 —— 冷启动场景下旧分数几乎必然是低分
    （被评的是「没有检索到」之类的兜底话术），会把联网搜到的好答案一并拦在库外。
    """
    new_message = "approve" if user_answer == "approve" else "rejected"
    values = {"human_answer": new_message}
    if new_message == "rejected":
        values["evaluate_score"] = None
    # 把人为输入存入图的 state 中
    graph.update_state(config=config, values=values)


def normalize_content(content) -> str:
    """把消息内容统一成字符串。

    AIMessage 的 content 可能是 str，也可能是多模态列表
    （[{"type": "text", "text": ...}, {"type": "image_url", ...}]），
    后者直接送进嵌入模型会出错，这里只取其中的文本片段。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content)


async def save_final_answer(state_values: dict) -> None:
    """把本轮最终回复写入上下文向量库（t_context_collection）。

    两个入口共用这一个函数，写入规则（来源标记、质量闸门）只维护一份。
    """
    messages = state_values.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        log.info("最后一条消息不是 AIMessage，跳过写入 Milvus")
        return

    answer = normalize_content(messages[-1].content)
    if not answer.strip():
        log.info("最终回复为空，跳过写入 Milvus")
        return

    # 来源标记：本轮轨迹里出现过 my_search 的 ToolMessage，即视为联网检索所得，
    # 与历史对话回答区分开，便于溯源。依据轨迹判断（而不是额外的 state 字段），
    # 因为 state 在同一 thread 内跨轮累积，用字段会把后续轮次一起误标。
    used_web_search = any(
        isinstance(m, ToolMessage) and m.name == "my_search" for m in messages
    )
    message_type = "WebSearch" if used_web_search else "AIMessage"

    log.info(f"开始写入Milvus（message_type={message_type}）")
    # 注意：async_insert 的形参名是 username（见 save_context.py），不是 user；
    # 状态里的用户名字段同样叫 username。
    # evaluate_score 一起传下去：写入器内部的质量闸门会据此拒绝低价值回答入库，
    # 防止「抱歉，没有检索到…」这类兜底话术被写回库、形成冷启动自锁。
    await get_milvus_writer().async_insert(
        context_text=answer,
        username=state_values.get("username", "ZS"),
        message_type=message_type,
        evaluate_score=state_values.get("evaluate_score"),
        # 本轮提问：写入器拿它当幂等键，同一个问题重复提问不会再落第二条。
        # input_text 由 process_input 在每轮开头写入，取到的即本轮提问；
        # 仅图片输入时为 None，此时写入器只做内容级去重。
        question=state_values.get("input_text"),
    )
