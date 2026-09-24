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
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore

from embedding.gme_qwen2_vl_2b_embedding import call_local_model
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
    """检索决策节点：判断是否调用 search_context 检索历史对话上下文。"""
    llm_with_tools = glm_llm_flash.bind_tools(context_tools)

    system_message = SystemMessage(content="""你是检索决策助手。你的唯一任务：判断是否需要调用 `search_context` 工具，检索该用户的历史对话上下文。

# 工具说明：
`search_context` 检索的是该用户的历史对话记录（对话记忆），返回过去对话中的相关片段，可能为空。

# 决策规则：
1. 消息中包含任何技术内容（技术概念、报错、配置、代码、命令、架构、实践方案等，不限技术领域）-> 必须调用 `search_context`，并把用户的问题提炼成简洁的检索关键词作为查询参数。
2. 消息同时包含寒暄和技术问题 -> 以技术问题为准，必须调用工具。
3. 消息是纯寒暄、问候、闲聊，或与技术无关的日常问题（如"你好""谢谢"）-> 不调用工具，直接简短回复。

你只做决策：该调工具就调工具，不该调就简短回复。永远不要在回复中尝试回答技术问题。""")

    reply = llm_with_tools.invoke([system_message, *state["messages"]])

    # 未触发工具调用 = 本轮是纯寒暄/闲聊直答。
    # 打上标记供 save_final_answer 跳过入库：闲聊回复对检索几乎零贡献，
    # 写进上下文库只会稀释检索结果（身份归属由 username 字段保证，不依赖闲聊内容）。
    if not getattr(reply, "tool_calls", None):
        return {"messages": [reply], "is_chitchat": True}

    return {"messages": [reply]}


def second_chatbot(state: MultimodalRAGState):
    """第二次生成回复：基于检索到的历史上下文作答（检索结果在 ToolMessage 里）。"""
    system_message = SystemMessage(content="""你是通用 AI 技术助理。请基于上方工具返回的历史对话内容，回答用户的技术问题。

# 回答规则（必须严格遵守）：
1. 回答必须完全基于工具返回的历史对话内容，严禁凭借自身内部知识补充技术细节。
2. 工具返回的内容无论以什么格式呈现（如 Markdown 片段、代码块），来源都是**该用户的历史对话记录**；说明来源时表述为"根据我们之前的对话"，不要说成"文档""资料"或"知识库文档"。
3. 如果工具明确返回"没有找到相关的历史上下文信息。"，统一回复："关于这个问题，我当前的历史对话记录中没有找到确切的资料。"
4. 回答使用 Markdown 格式。""")

    return {"messages": [glm_llm_flash.invoke([system_message, *state["messages"]])]}


def _format_hit_time(ts) -> str:
    """把库里的毫秒时间戳格式化成可读时间；缺失或异常时返回空串。"""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(int(ts) / 1000))
    except (TypeError, ValueError, OSError):
        return ""


def third_chatbot(state: MultimodalRAGState):
    """第三次生成回复：基于 retriever_node 检索到的历史对话上下文作答。"""
    # 从向量数据库中检索到的历史对话片段（纯文本，对话库里没有图片字段）
    context_retrieved = state.get("context_retrieved")

    # 注意：retriever_node 检索的是 t_context_collection（历史对话上下文库），
    # 返回字段为 context_text / username / timestamp / message_type，
    # 不存在 text / filename / image_path（那是文档库 t_doc_collection 的字段），切勿混用。
    count = 0
    context_pieces = []
    for hit in (context_retrieved or []):
        count += 1
        content = hit.get("context_text") or ""
        when = _format_hit_time(hit.get("timestamp"))
        # 不把 username 当"资料来源"：它是人不是资料，露给模型后容易被写进回答
        # （出现"来源：张三"这种既无用又泄露用户名的表述）。改用时间戳做片段标识。
        header = f"片段{count}" + (f"（{when}）" if when else "")
        context_pieces.append(f"{header}：\n{content}")
    context = "\n\n".join(context_pieces) if context_pieces else "（本轮未检索到相关片段）"

    input_text = state.get("input_text")
    input_image = state.get("input_image")

    system_prompt = f"""你是通用 AI 技术助理。请严格依据下方「历史对话上下文」回答用户的问题。

# 上下文说明
- 下方内容是检索出的、该用户历史对话中的片段，可能不完整，也可能为空。
- 这些内容的唯一来源是「该用户的历史对话记录」。说明来源时统一表述为「根据我们之前的对话」，
  不要称为「文档」「资料」「知识库」「上传的文件」。

# 回答要求
1. 回答必须完全基于下方上下文，严禁用你自己的内部知识补充技术细节。
2. 若上下文为空、或与用户的问题无关，直接回复下面这句话，不要另行作答：
   「关于这个问题，我当前的历史对话记录中没有找到确切的资料。」
3. 输出使用 Markdown 格式。
4. 回答的最后单独一行标注来源，格式固定为：> 来源：历史对话记录
5. 若用户本次还提供了文本或图片，且与上下文相关，可结合上下文一起作答；不要脱离上下文自由发挥。

# 历史对话上下文
{context}"""

    # 构建用户消息内容
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})

    # 这里刻意不用 ChatPromptTemplate：历史对话内容里出现花括号很常见（JSON、代码块），
    # 一旦被当成模板字符串，花括号会被解析成变量占位符并抛 "missing variables" 错。
    # 直接构造消息对象，就不存在二次模板解析这一步。
    messages = [SystemMessage(content=system_prompt)]
    if user_content:
        messages.append(HumanMessage(content=user_content))

    return {"messages": [glm_llm_flash.invoke(messages)]}


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

    system_message = SystemMessage(content="""你是联网检索助手。当本地历史对话不足以回答用户的问题时，由你联网查找资料作答。

# 工作方式
1. 检索阶段：如果历史消息中还没有 `my_search` 的返回结果，调用一次 `my_search`，query 填写用户本轮问题的核心关键词。
2. 作答阶段：如果历史消息中已经出现 `my_search` 的返回结果，直接基于结果作答，**不得再次调用工具**。

# 回答规则
1. 回答必须基于搜索结果，严禁使用你自己的内部知识补充搜索结果之外的技术细节。
2. 使用 Markdown 格式，并在开头说明「以下内容来自网络搜索」。
3. 引用来源时只使用搜索结果中实际出现的信息；搜索结果没有给出链接时，不要编造链接、网址或文献出处。
4. 若工具返回「没有搜索到任何内容！」，说明检索失败，请如实回复「未找到相关网络资料」，不要编造。
5. 若用户本次还提供了文本或图片，请结合它们理解问题后再作答。""")

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
    # 纯寒暄/闲聊轮（first_chatbot 直答、未调工具）：不入库，避免低价值内容稀释检索
    if state_values.get("is_chitchat"):
        log.info("本轮为寒暄/闲聊直答，跳过写入 Milvus")
        return

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


def warm_up_embedding() -> None:
    """启动时预热嵌入模型。

    模型权重首次加载的开销只在「第一次编码」时发生。不预热的话，这笔开销会落在
    用户的第一句话上（实测一次首轮 53s，其中约 19s 纯粹是加载权重）；
    预热后它被挪到服务启动阶段，此后每次编码只需几十毫秒。
    具体数值随模型规格与设备而变，这里只作量级参考。
    """
    try:
        t0 = time.time()
        log.info("开始预热嵌入模型（首次加载权重）…")
        call_local_model([{"text": "预热"}])
        log.info(f"嵌入模型预热完成，耗时 {time.time() - t0:.2f}s")
    except Exception as e:
        # 预热只是优化，失败不影响启动：真正的加载会退回到首次编码时进行
        log.exception(f"嵌入模型预热失败（不影响启动，首次编码时会重试）: {e}")