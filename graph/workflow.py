import asyncio
import os
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.store.memory import InMemoryStore

from embedding.gme_qwen2_vl_2b_embedding import image_to_base64
from graph.all_router import route_only_image, route_llm_or_retriever, route_evaluate_node, route_human_node, \
    route_human_approval_node
from graph.common import InvalidInputError
from graph.custom_state import MultimodalRAGState
from graph.evaluate_node import evaluate_answer
from graph.print_messages import pretty_print_messages
from graph.save_context import get_milvus_writer
from graph.search_node import SearchContextToolNode, retriever_node
from graph.tools import search_context, my_search
from models.init_chat_model_llm import glm_llm_flash
from utils.log_utils import log

# 上下文检索工具列表
tools = [search_context]
# 网络搜索 my_search 由确定性节点 web_search_node 调用，不再走「模型自主决定是否调工具」的模式


# ============== 工作流节点函数 ==============
def process_input(state: MultimodalRAGState, config: RunnableConfig):
    """
    处理用户输入
    Args:
        state (MultimodalRAGState): 状态对象,根据用户输入的类型,更新状态对象的属性
        config (RunnableConfig): 配置对象
    Returns:
        MultimodalRAGState: 更新后的状态对象
    """

    # 从配置对象中获取用户名
    username = config["configurable"].get("username", "ZS")
    # 从状态对象中获取最后一个消息, HumanMessage 类型，用户输入的消息
    last_message = state["messages"][-1]

    log.info(f"用户{username}输入: {last_message.content}")

    input_type = "has_text"
    text_content = None
    image_url = None

    # 检查输入的类型
    if isinstance(last_message, HumanMessage):
        # 多模态的消息，包含文本和图像
        if isinstance(last_message.content, list):
            content = last_message.content
            print(f"多模态输入: {content}")
            for item in content:
                # 提取文本内容
                if item.get("type") == "text":
                    text_content = item.get("text", None)
                # 提取图像URL
                elif item.get("type") == "image_url":
                    url = item.get("image_url", "").get("url")
                    # 确保URL有效  （是图片的base64格式的字符串） （在线url）
                    if url:
                        image_url = url
    else:
        raise InvalidInputError(f"用户输入的消息错误！原始输入：{last_message}")

    if not text_content and image_url:
        input_type = "only_image"

    # 返回结果： 如果想把什么样的数据保存（更新）到状态中，返回一个字典，键为状态字段名称，值为数据。
    return {
        "input_type": input_type,
        "username": username,
        "input_text": text_content,
        "input_image": image_url,
    }


# 第一次生成回复或者决策（基于当前会话生成回复）
def first_chatbot(state: MultimodalRAGState):
    llm_with_tools = glm_llm_flash.bind_tools(tools)

    # 系统提示词示例
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

    # system_message = SystemMessage(
    #     content='你是一个Flink分布式计算引擎的专家，如果输入给你的信息中包含相关内容，直接回答。否则不要自己直接回答用户的问题，一定调用工具和知识库来补充，生成最终的答案。')

    return {"messages": [llm_with_tools.invoke([*state["messages"], system_message])]}


# 第二次生成回复（基于检索历史上下文 生成回复, 检索到的历史上下文在ToolMessage里面）
def second_chatbot(state: MultimodalRAGState):
    """
    第二次生成回复（基于检索历史上下文 生成回复, 检索到的历史上下文在ToolMessage里面）
    Args:
        state (MultimodalRAGState): 状态对象
    Returns:
        MultimodalRAGState: 更新后的状态对象
    """
    return {"messages": [glm_llm_flash.invoke(state["messages"])]}


# 第三次 生成回复（基于检索到的历史对话上下文 生成回复, 检索到的结果在状态里面）
def third_chatbot(state: MultimodalRAGState):
    """
    处理多模态请求并返回 Markdown 格式的结果
    Args:
        state (MultimodalRAGState): 状态对象
    Returns:
        MultimodalRAGState: 更新后的状态对象
    """

    # 从向量数据库中检索到的文本内容
    context_retrieved = state.get('context_retrieved')
    # 从向量数据库中检索到的图像URL 路径
    image_retrieved = state.get('image_retrieved')

    # 处理上下文列表
    # 注意：retriever_node 检索的是 t_context_collection（历史对话上下文库），
    # 返回字段为 context_text / username / timestamp / message_type，
    # 不存在 text / filename（那是文档库 t_doc_collection 的字段），切勿混用。
    count = 0
    context_pieces = []
    for hit in (context_retrieved or []):
        count += 1
        content = hit.get('context_text')
        source = hit.get('username') or '未知来源'
        context_pieces.append(f"检索后的内容{count}：\n {content} \n 资料来源：{source}")
    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息。"

    inpt_text = state.get('input_text')
    input_image = state.get('input_image')
    # 构建系统提示词
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
    if inpt_text:
        user_content.append({"type": "text", "text": inpt_text})
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
    """确定性网络搜索节点：本地上下文库没有可用内容时，直接联网检索。

    改为确定性触发（不依赖大模型是否"愿意"调用工具），与 workflow_gradio.py 保持一致。
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
    # 来源标记挂在消息上（而不是 state 字段）：state 在同一 thread 内会跨轮累积，
    # 用字段会导致后续轮次被误标为联网来源。
    answer.additional_kwargs["answer_source"] = "web_search"
    return {"messages": [answer]}

    # =======================创建工作流=======================


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
builder.add_node("fourth_chatbot", fourth_chatbot)
builder.add_node("web_search_node", web_search_node)

# 添加边
builder.add_edge(START, 'process_input')
builder.add_conditional_edges("process_input", route_only_image,
                              {"retriever_node": "retriever_node", 'first_chatbot': 'first_chatbot'})

builder.add_conditional_edges("first_chatbot", tools_condition, {"tools": "search_context", END: END})

builder.add_conditional_edges("search_context", route_llm_or_retriever,
                              {"web_search_node": "web_search_node", 'second_chatbot': 'second_chatbot'})

builder.add_edge('retriever_node', 'third_chatbot')
builder.add_conditional_edges("third_chatbot", route_evaluate_node, {"evaluate_node": "evaluate_node",END:END})
builder.add_conditional_edges('evaluate_node', route_human_node, {"human_approval": "human_approval", END: END})
# 人工审批拒绝后：先执行网络搜索（web_search_node），再由大模型基于搜索结果作答，最后自动写回向量库
builder.add_conditional_edges('human_approval', route_human_approval_node, {"fourth_chatbot": "web_search_node", END: END})

builder.add_edge('web_search_node', 'fourth_chatbot')

graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
    interrupt_before=['human_approval']  # 添加中断点   静态的人工介入， 当恢复工作流时，会从中断点开始恢复工作流
)

mermaid_code = graph.get_graph().draw_mermaid_png()
# 按脚本自身所在目录写出，避免随工作目录漂移（此前从项目根运行会把图写到根目录）
_png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_rag.png")
with open(_png_path, "wb") as f:
    f.write(mermaid_code)

session_id = str(uuid.uuid4())

# 配置参数，包含用户名和线程ID
config = {
    "configurable": {
        # 键名必须与 process_input 读取的键一致（config["configurable"].get("username")）
        "username": "ZS",
        # 检查点由session_id访问
        "thread_id": session_id,
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


def _normalize_content(content) -> str:
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


async def _save_final_answer(current_state) -> None:
    """把工作流产出的最终 AI 回复写入上下文向量库（t_context_collection）。"""
    mess = current_state.values.get('messages', [])
    if not mess or not isinstance(mess[-1], AIMessage):
        log.info("最后一条消息不是 AIMessage，跳过写入 Milvus")
        return

    answer = _normalize_content(mess[-1].content)
    if not answer.strip():
        log.info("最终回复为空，跳过写入 Milvus")
        return

    # 来源标记：联网检索得到的知识用 WebSearch 标出，与历史对话记录区分，便于溯源
    src = mess[-1].additional_kwargs.get('answer_source')
    message_type = "WebSearch" if src == 'web_search' else "AIMessage"

    log.info(f"开始写入Milvus（message_type={message_type}）")
    # 注意：async_insert 的形参名是 username（见 save_context.py），不是 user；
    # 状态里的用户名字段同样叫 username。
    # evaluate_score 一起传下去：写入器内部的质量闸门会据此拒绝低价值回答入库，
    # 防止「抱歉，没有检索到…」这类兜底话术被写回库、形成冷启动自锁。
    await get_milvus_writer().async_insert(
        context_text=answer,
        username=current_state.values.get('username', 'ZS'),
        message_type=message_type,
        evaluate_score=current_state.values.get('evaluate_score'),
    )


async def execute_graph(user_input: str) -> str:
    """ 执行工作流的函数"""
    result = ''  # AI助手的最后一条消息
    current_state = graph.get_state(config)  # 得到实时的状态（短期上下文）
    if current_state.next:  # 出现了工作流的中断
        # 通过提供关于请求的更改/改变主意的指示来满足图的继续执行
        update_state(user_input, config)
        # 恢复执行工作流
        async for chunk in graph.astream(None, config, stream_mode='values'):
            pretty_print_messages(chunk, last_message=True)
    else:
        image_base64 = None
        text = None
        if '&' in user_input:
            text = user_input.split('&')[0]
            image = user_input.split('&')[1]
            if image and os.path.isfile(image):
                image_base64 = {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image)[0]},
                }
        elif os.path.isfile(user_input):
            image_base64 = {
                "type": "image_url",
                "image_url": {"url": image_to_base64(user_input)[0]},
            }
        else:
            text = user_input

        message = HumanMessage(
            content=[
            ]
        )
        if text:
            message.content.append({"type": "text", "text": text})
        if image_base64:
            message.content.append(image_base64)
        async for chunk in graph.astream({'messages': [message]}, config, stream_mode='values'):
            pretty_print_messages(chunk, last_message=True)

    # 收尾处理：两条路径（新输入 / 中断恢复）统一在这里判断，
    # 不能再在中断分支里提前 return，否则会跳过下面的写库逻辑。
    current_state = graph.get_state(config)
    if current_state.next:  # 仍在中断点（等待人工审批）
        output = ("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
                  "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！")
        result = output
    else:
        # 写入响应到Milvus（把当前工作流执行后的最终结果，保存到上下文的向量数据库中）
        await _save_final_answer(current_state)

    return result


async def main():
    # 执行工作流
    while True:
        user_input = input('用户输入(文本和图片用&隔开)：')
        if user_input.lower() in ['exit', 'quit', '退出']:
            break

        res = await execute_graph(user_input)
        if res:
            print('AI: ', res)


if __name__ == '__main__':
    asyncio.run(main())