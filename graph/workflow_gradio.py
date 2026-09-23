import os
import uuid
from typing import List, Dict

import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.store.memory import InMemoryStore

from embedding.gme_qwen2_vl_2b_embedding import image_to_base64
from graph.all_router import route_only_image, route_llm_or_retriever, route_evaluate_node, route_human_node, \
    route_human_approval_node
from graph.common import InvalidInputError
from graph.custom_state import MultimodalRAGState
from graph.evaluate_node import evaluate_answer
from graph.save_context import get_milvus_writer
from graph.search_node import SearchContextToolNode, retriever_node
from graph.tools import search_context, my_search
from models.init_chat_model_llm import glm_llm_flash
from utils.log_utils import log

# 上下文检索工具列表
tools = [search_context]


def _role(message) -> str:
    """兼容 dict 与 gradio ChatMessage，取 role"""
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _get_content(message):
    """兼容 dict 与 gradio ChatMessage，取 content"""
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _extract_text_from_content(content) -> str:
    """从 content 中提取纯文本（兼容 str / list 两种形态）"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return ""


# 工作流节点函数
def process_input(state: MultimodalRAGState, config: RunnableConfig):
    """处理用户输入"""
    user_name = config['configurable'].get('username', 'ZS')
    last_message = state["messages"][-1]
    log.info(f"用户 {user_name} 输入：{last_message}")
    input_type = 'has_text'
    text_content = None
    image_url = None
    # 检查输入类型
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):  # 多模态的消息
            content = last_message.content
            for item in content:
                # 提取文本内容
                if item.get("type") == "text":
                    text_content = item.get("text", None)

                # 提取图片URL
                elif item.get("type") == "image_url":
                    url = item.get("image_url", "").get('url')
                    if url:  # 确保URL有效  （是图片的base64格式的字符串） （在线url）
                        image_url = url
    else:
        raise InvalidInputError(f"用户输入的消息错误！原始输入：{last_message}")

    if not text_content and image_url:
        input_type = 'only_image'

    # 返回结果： 如果想把什么样的数据保存（更新）到状态中，返回一个字典，键为状态字段名称，值为数据。
    return {"input_type": input_type, 'username': user_name, 'input_text': text_content, 'input_image': image_url}


# 第一次生成回复或者决策（基于当前会话生成回复）
def first_chatbot(state: MultimodalRAGState):
    llm_with_tools = glm_llm_flash.bind_tools(tools)
    system_message = SystemMessage(content="""你是一名专精于 Apache Flink 的 AI 助手，可以调用工具调取与该用户的历史对话记录。

    # 工具说明（务必准确理解）：
    `search_context` 检索的是**该用户的历史对话上下文**（对话记忆），不是 Flink 文档知识库；
    返回内容为过去对话中的相关片段，可能为空。

    # 核心指令（必须严格遵守）：
    1.  **首要规则**：当用户提问涉及 Apache Flink 的任何技术概念、配置、代码或实践时，你**必须且只能**调用 `search_context` 工具来获取信息。
    2.  **禁止行为**：你**严禁**凭借自身内部知识直接回答任何 Flink 技术问题。你的回答必须完全基于工具返回的历史对话内容。
    3.  **表述要求**：工具返回的是历史对话记录，不要把它描述成“文档”“资料”或“知识库文档”。
    4.  **兜底策略**：如果工具返回了相关信息，请基于这些信息组织答案。如果工具明确返回“没有找到相关的历史上下文信息。”，你应统一回复：“关于这个问题，我当前的历史对话记录中没有找到确切的资料。”

    # 回答流程（不可更改）：
    用户提问 -> 调用 `search_context` 工具 -> 基于工具返回结果生成答案。
    """)

    return {"messages": [llm_with_tools.invoke([*state["messages"], system_message])]}


# 第二次生成回复（基于检索历史上下文 生成回复, 检索到的历史上下文在ToolMessage里面）
def second_chatbot(state: MultimodalRAGState):
    return {"messages": [glm_llm_flash.invoke(state["messages"])]}


# 第三次 生成回复（基于检索到的历史对话上下文 生成回复, 检索到的结果在状态里面）
def third_chatbot(state: MultimodalRAGState):
    """处理多模态请求并返回Markdown格式的结果"""
    context_retrieved = state.get('context_retrieved') or []
    images = state.get('image_retrieved') or []

    # 处理上下文列表
    # 注意：检索结果来自 t_context_collection（历史对话上下文库），
    # 字段为 context_text / username / timestamp / message_type，
    # 不存在 text / filename（那是文档库 t_doc_collection 的字段），切勿混用。
    count = 0
    context_pieces = []
    for hit in context_retrieved:
        count += 1
        content = hit.get('context_text')
        source = hit.get('username') or '未知来源'
        context_pieces.append(f"检索后的内容{count}：\n {content} \n 资料来源：{source}")

    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息。"

    input_text = state.get('input_text')
    input_image = state.get('input_image')

    # 构建系统提示词
    system_prompt = f"""
        请根据用户输入和以下检索到的「历史对话上下文」生成响应。
        注意：这些内容是从该用户的历史对话记录中检索出的片段，不是文档或知识库资料。
        如果上下文内容中没有相关答案，请直接说明，不要自己直接输出答案。
        要求：
        1. 响应必须使用Markdown格式
        2. 在响应文字下方显示所有相关图片，图片的路径列表为{images}，使用Markdown图片语法：
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
            ("user", user_content)
        ]
    )

    chain = prompt | glm_llm_flash

    return {"messages": [chain.invoke({'context': context})]}


def human_approval(state: MultimodalRAGState):
    log.info('已经进入了人工审批节点')
    log.info(f'当前的状态中的人工审批信息：{state["human_answer"]}')


def web_search_node(state: MultimodalRAGState):
    """直接执行网络搜索，把结果写入状态字段 web_search_result。

    确定性触发：不依赖大模型是否"愿意"调用工具，保证被拒绝的答案一定会触发联网搜索。
    """
    query = state.get('input_text')
    if not query:
        # 兜底：取最近一条用户文本消息
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content if isinstance(msg.content, str) else None
                break
    query = (query or "").strip()
    log.info(f"开始执行网络搜索：{query}")
    if query:
        result = my_search.invoke({"query": query})
    else:
        result = "没有搜索到任何内容！"
    return {"web_search_result": str(result)}


def fourth_chatbot(state: MultimodalRAGState):
    """基于网络搜索结果生成最终回复（web_search_node 已把结果写入 web_search_result）"""
    search_result = state.get('web_search_result') or ""
    system_message = SystemMessage(content=(
        "你是一个智能体助手，请严格基于下面的【网络搜索结果】回答用户的问题。\n"
        "要求：\n"
        "1. 使用 Markdown 格式作答，尽量标注信息来源。\n"
        "2. 若【网络搜索结果】中没有相关内容，请如实说明「未找到相关网络资料」，不要编造。\n\n"
        "【网络搜索结果】\n" + search_result
    ))
    # 保留完整对话历史（含用户提问、历史对话回答），供模型结合搜索结果组织最终回复
    history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    answer = glm_llm_flash.invoke([system_message, *history])
    # 来源标记挂在消息上（而不是 state 字段）：state 在同一 thread 内会跨轮累积
    answer.additional_kwargs["answer_source"] = "web_search"
    return {"messages": [answer]}


checkpointer = InMemorySaver()
store = InMemoryStore()

# 创建图
builder = StateGraph(MultimodalRAGState)

# 添加节点
builder.add_node("process_input", process_input)
builder.add_node("first_chatbot", first_chatbot)

search_context_node = SearchContextToolNode(tools=tools)
builder.add_node("search_context", search_context_node)
builder.add_node("retriever_node", retriever_node)
builder.add_node("second_chatbot", second_chatbot)
builder.add_node("third_chatbot", third_chatbot)
builder.add_node("evaluate_node", evaluate_answer)
builder.add_node("human_approval", human_approval)
builder.add_node("web_search_node", web_search_node)
builder.add_node("fourth_chatbot", fourth_chatbot)

# 添加边
builder.add_edge(START, 'process_input')
builder.add_conditional_edges('process_input', route_only_image,
                              {"retriever_node": "retriever_node", 'first_chatbot': 'first_chatbot'})

builder.add_conditional_edges('first_chatbot', tools_condition, {"tools": "search_context", END: END}, )

builder.add_conditional_edges('search_context', route_llm_or_retriever,
                              {"web_search_node": "web_search_node", 'second_chatbot': 'second_chatbot'})

builder.add_edge('retriever_node', 'third_chatbot')

# 命中历史上下文的回答，也要继续做评估，分数不合格时走人工审批 + 联网兜底
builder.add_edge('second_chatbot', 'evaluate_node')

builder.add_conditional_edges('third_chatbot', route_evaluate_node, {"evaluate_node": "evaluate_node", END: END}, )
builder.add_conditional_edges('evaluate_node', route_human_node, {"human_approval": "human_approval", END: END}, )
# 人工审批拒绝后：先执行网络搜索（web_search_node），再由大模型基于搜索结果生成最终回复
builder.add_conditional_edges('human_approval', route_human_approval_node,
                              {"fourth_chatbot": "web_search_node", END: END}, )
builder.add_edge('web_search_node', 'fourth_chatbot')

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=['human_approval']  # 添加中断点   静态的人工介入， 当恢复工作流时，会从中断点开始恢复工作流
)

# ============================================================
# 全局变量：当前活跃会话的 config
# 说明：
#   - 每次发起「新问题」时，会为本次会话生成新的 thread_id，并覆盖此变量；
#   - 下次提交时，先查这个 config 上是否有中断；
#   - 如果有，说明用户是在「回复审批」（approve/rejected），走恢复分支；
#   - 如果没有，说明是新问题，再生成新 thread_id。
# 为什么不直接用固定的 config？
#   如果 thread_id 固定，多次请求会互相污染检查点状态，导致中断无法正确恢复。
# ============================================================
current_run_config = {
    "configurable": {
        "username": "ZS",
        "thread_id": str(uuid.uuid4()),
    }
}


def update_state(user_answer, config):
    """在工作流外面的普通函数中，让人工介入"""
    if user_answer == 'approve':
        new_message = "approve"
    else:
        new_message = "rejected"
    # 把人为输入的，存入图的state中
    graph.update_state(
        config=config,
        values={'human_answer': new_message}
    )


def transcribe_image(image_path):
    """
    将本地图片转换为 base64 格式的 data URL 消息块
    :param image_path: 本地图片路径
    :return: image_url 消息块字典；转换失败返回 None
    """
    data_url, _ = image_to_base64(image_path)  # 已经是 "data:{mime};base64,xxxx" 完整格式
    if not data_url:
        return None
    return {
        "type": "image_url",
        "image_url": {"url": data_url},
    }


def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置,并返回后面的所有user消息"""
    if not history:
        return None
    if _role(history[-1]) == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if _role(history[i]) == "assistant":
            last_assistant_idx = i
            break

    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后查找第一个user
        return history[last_assistant_idx + 1:]


# 定义处理提交事件的函数
def add_message(history, user_input):
    """将用户输入的消息添加到聊天记录中"""
    if user_input['text'] is not None:  # 文本消息
        history.append({'role': 'user', "content": user_input['text']})

    for m in user_input['files']:
        print(m)
        history.append({'role': 'user', "content": {'path': m}})

    # 返回更新后的聊天历史记录和一个清空且不可交互的输入框
    return history, gr.MultimodalTextbox(value=None, interactive=False)


async def submit_llm(history: List[Dict]):
    """把用户的输入，提交给工作流（大模型）处理，并流式渲染结果"""
    global current_run_config

    # 用「当前活跃会话的 config」查询中断状态
    current_state = graph.get_state(current_run_config)
    inputs = None

    if current_state.next:
        # -------- 情况 A：当前会话有中断，用户可能在回复审批 --------
        raw_answer = _get_content(history[-1]) if history else ""
        user_answer = _extract_text_from_content(raw_answer)

        if user_answer in ('approve', 'rejected'):
            # A1. 用户确实是回复审批 → 恢复中断
            update_state(user_answer, current_run_config)
            run_config = current_run_config
            log.info(f"[resume] 用户答复={user_answer}，恢复中断")
        else:
            # A2. 用户发了新问题 → 放弃旧中断，开启新会话
            log.info(f"[new] 检测到旧中断未回复，用户输入新问题，开启新会话")
            current_run_config = {
                "configurable": {
                    "username": "ZS",
                    "thread_id": str(uuid.uuid4()),
                }
            }
            run_config = current_run_config
            inputs = _build_inputs(history)
            if inputs is None:
                history.append(ChatMessage(role="assistant", content="⚠️ 没有解析到有效输入，请重新输入。"))
                yield history
                return
    else:
        # -------- 情况 B：没有中断 → 新问题 --------
        current_run_config = {
            "configurable": {
                "username": "ZS",
                "thread_id": str(uuid.uuid4()),
            }
        }
        run_config = current_run_config
        inputs = _build_inputs(history)
        if inputs is None:
            history.append(ChatMessage(role="assistant", content="⚠️ 没有解析到有效输入，请重新输入。"))
            yield history
            return

    # ---- 执行工作流 ----
    full_response = ""

    async for chunk in graph.astream(
            inputs,
            run_config,
            stream_mode=["messages", "updates"],
    ):
        if not isinstance(chunk, tuple):
            continue
        mode, payload = chunk

        # ---- messages 流：数据为 (AIMessageChunk, metadata) ----
        if mode == "messages":
            if not isinstance(payload, tuple) or len(payload) != 2:
                continue
            msg, _meta = payload
            if isinstance(msg, AIMessage) and msg.content:
                full_response += msg.content
                if (history and isinstance(history[-1], dict)
                        and history[-1].get("role") == "assistant"
                        and not history[-1].get("metadata", {}).get("title")):
                    history[-1]["content"] = full_response
                else:
                    history.append({"role": "assistant", "content": full_response})
                yield history
            continue

        # ---- updates 流：数据为 {节点名: 该节点返回的更新字典} ----
        if mode == "updates":
            for node, update in payload.items():
                if not isinstance(update, dict):
                    continue

                # 网络搜索节点：展示工具调用
                if "web_search_result" in update:
                    history.append(ChatMessage(
                        role="assistant",
                        content="🔧 已调用网络搜索，正在基于搜索结果生成回复...",
                        metadata={"title": "🛠️ 工具调用: 互联网搜索"},
                    ))
                    full_response = ""
                    yield history

                # 输出 ToolMessage 的工具节点（如 search_context）
                for message in update.get("messages", []):
                    if isinstance(message, ToolMessage):
                        tool_msg = f"🔧 已调用工具 `{message.name}`：\n{str(message.content)[:300]}"
                        history.append(ChatMessage(
                            role="assistant",
                            content=tool_msg,
                            metadata={"title": f"🛠️ 工具调用: {message.name}"},
                        ))
                        full_response = ""
                        yield history

    # ---- 检查工作流是否又发生了中断（人工审批点）----
    current_state = graph.get_state(run_config)
    if current_state.next:
        output = ("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
                  "如果认可，请输入「approve」，否则请输入「rejected」，系统将调用网络搜索引擎重新生成回复！")
        history.append(ChatMessage(role="assistant", content=output))
        yield history
    else:
        # 写入响应到 Milvus（最终答案存入历史对话上下文库）
        mess = current_state.values.get('messages', [])
        if mess and isinstance(mess[-1], AIMessage):
            # content 可能是多模态列表，先归一化成纯文本再向量化
            answer = _extract_text_from_content(mess[-1].content)
            if answer:
                # 来源标记：联网检索得到的知识用 WebSearch 标出，与历史对话记录区分，便于溯源
                src = mess[-1].additional_kwargs.get('answer_source')
                message_type = "WebSearch" if src == 'web_search' else "AIMessage"
                log.info(f"开始写入Milvus（message_type={message_type}）")
                # 必须 await：若丢进后台任务而不等待，任务可能随本次请求结束被取消，写库会静默丢失
                # evaluate_score 一起传下去：写入器内部的质量闸门会据此拒绝低价值回答入库，
                # 防止「抱歉，没有检索到…」这类兜底话术被写回库、形成冷启动自锁。
                await get_milvus_writer().async_insert(
                    context_text=answer,
                    username=current_state.values.get('username', 'ZS'),
                    message_type=message_type,
                    evaluate_score=current_state.values.get('evaluate_score'),
                )


def _build_inputs(history: List[Dict]):
    """从聊天记录中提取用户最新的输入，构建工作流的 inputs。

    Returns:
        dict 形如 {'messages': [HumanMessage]}，若无有效内容返回 None。
    """
    user_input_messages = get_last_user_after_assistant(history)
    log.info(f"本次需要处理的消息：{user_input_messages}")

    content = []
    if user_input_messages:
        for x in user_input_messages:
            msg_content = _get_content(x)

            # ① 字符串：直接文本
            if isinstance(msg_content, str) and msg_content.strip():
                content.append({'type': 'text', 'text': msg_content})
                continue

            # ② Gradio 6 多模态 list：可能包含 text / image_url
            if isinstance(msg_content, list):
                for item in msg_content:
                    if not isinstance(item, dict):
                        continue
                    if item.get('type') == 'text':
                        text = item.get('text', "")
                        if text and text.strip():
                            content.append({'type': 'text', 'text': text})
                    elif item.get('type') == 'image_url':
                        content.append(item)
                continue

            # ③ 文件路径：tuple 或 dict
            file_path = None
            if isinstance(msg_content, tuple):
                file_path = msg_content[0]
            elif isinstance(msg_content, dict):
                file_path = msg_content.get('path') or msg_content.get('url')
            if file_path and os.path.isfile(file_path):
                file_message = transcribe_image(file_path)
                if file_message:
                    content.append(file_message)

    if not content:
        return None

    input_message = HumanMessage(content=content)
    return {'messages': [input_message]}


css = '''
#bgc {background-color: #7FFFD4}
.feedback textarea {font-size: 24px !important}
.message { font-family: monospace; }
.gradio-container { font-family: monospace; }
'''

with gr.Blocks(title='多模态RAG项目') as instance:
    gr.Label('多模态RAG项目', container=False)

    chatbot = gr.Chatbot(
        height=450,
        label='AI助手',
        render_markdown=True,   # 启用Markdown渲染
        line_breaks=False,      # 禁用自动换行符
        elem_id="bgc"           # 让 #bgc 样式生效
    )  # 聊天记录组件

    # 多模态输入框
    chat_input = gr.MultimodalTextbox(
        file_types=['png'],
        file_count='multiple',
        placeholder='请输入文字或者图片信息...',
        show_label=False,
        sources=['upload'],
    )

    chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input]
    ).then(
        submit_llm,
        [chatbot],
        [chatbot],
    ).then(  # 回复完成后重新激活输入框
        lambda: gr.MultimodalTextbox(interactive=True),  # 匿名函数重置输入框
        None,  # 无输入
        [chat_input]  # 输出到输入框
    )

if __name__ == '__main__':
    # 启动Gradio的应用
    instance.launch(
        debug=True,
        theme=gr.themes.Soft(),
        css=css
    )