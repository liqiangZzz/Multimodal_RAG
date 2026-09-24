"""多轮对话 RAG 工作流的 Gradio 界面入口。

节点函数与图结构全部在 graph.graph_builder，本文件只保留界面与流式渲染外壳，
与 workflow.py 共用同一份图，两者不会再有流程差异。

运行：

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m graph.workflow_gradio
"""

import os
from typing import Dict, List

import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from embedding.gme_qwen2_vl_2b_embedding import image_to_base64
from graph.graph_builder import graph, new_run_config, save_final_answer, update_state, warm_up_embedding
from utils.log_utils import log


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
current_run_config = new_run_config()


def transcribe_image(image_path):
    """将本地图片转换为 base64 格式的 data URL 消息块"""
    data_url, _ = image_to_base64(image_path)  # 已经是 "data:{mime};base64,xxxx" 完整格式
    if not data_url:
        return None
    return {
        "type": "image_url",
        "image_url": {"url": data_url},
    }


def get_last_user_after_assistant(history):
    """反向遍历找到最后一个 assistant 的位置，并返回后面的所有 user 消息"""
    if not history:
        return None
    if _role(history[-1]) == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if _role(history[i]) == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx == -1:
        return history
    return history[last_assistant_idx + 1:]


def add_message(history, user_input):
    """将用户输入的消息添加到聊天记录中"""
    if user_input['text'] is not None:  # 文本消息
        history.append({'role': 'user', "content": user_input['text']})

    for m in user_input['files']:
        print(m)
        history.append({'role': 'user', "content": {'path': m}})

    # 返回更新后的聊天历史记录和一个清空且不可交互的输入框
    return history, gr.MultimodalTextbox(value=None, interactive=False)


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

            # ② Gradio 多模态 list：可能包含 text / image_url
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


async def submit_llm(history: List[Dict]):
    """把用户的输入提交给工作流处理，并流式渲染结果"""
    global current_run_config

    # 用「当前活跃会话的 config」查询中断状态
    current_state = graph.get_state(current_run_config)
    inputs = None

    if current_state.next:
        # -------- 情况 A：当前会话有中断，用户可能在回复审批 --------
        raw_answer = _get_content(history[-1]) if history else ""
        user_answer = _extract_text_from_content(raw_answer)

        if user_answer in ('approve', 'rejected'):
            # A1. 用户确实是回复审批 -> 恢复中断
            update_state(user_answer, current_run_config)
            run_config = current_run_config
            log.info(f"[resume] 用户答复={user_answer}，恢复中断")
        else:
            # A2. 用户发了新问题 -> 放弃旧中断，开启新会话
            log.info("[new] 检测到旧中断未回复，用户输入新问题，开启新会话")
            current_run_config = new_run_config()
            run_config = current_run_config
            inputs = _build_inputs(history)
            if inputs is None:
                history.append(ChatMessage(role="assistant", content="⚠️ 没有解析到有效输入，请重新输入。"))
                yield history
                return
    else:
        # -------- 情况 B：没有中断 -> 新问题 --------
        current_run_config = new_run_config()
        run_config = current_run_config
        inputs = _build_inputs(history)
        if inputs is None:
            history.append(ChatMessage(role="assistant", content="⚠️ 没有解析到有效输入，请重新输入。"))
            yield history
            return

    # ---- 执行工作流 ----
    # 从提交到第一个 token 之间有一段空档：工具节点要做嵌入编码、混合检索，
    # 再等外部评委打分。这期间前端没有增量输出、输入框又被禁用，
    # 看起来和"卡死"没有区别（2026-09-23 曾因此误判为阻塞）。
    # 先放一条占位提示，第一个 token 到达时会被原地覆盖。
    _PLACEHOLDER = "🔍 正在检索历史对话上下文，请稍候…"
    history.append({"role": "assistant", "content": _PLACEHOLDER})
    yield history

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
            # 只追加模型输出（AIMessageChunk 是 AIMessage 的子类）；
            # stream_mode='messages' 也会带出 ToolMessage 等非模型消息，
            # 若不过滤会把工具返回内容混进回答气泡里。
            if isinstance(msg, AIMessage) and msg.content:
                full_response += msg.content
                if (history and isinstance(history[-1], dict)
                        and history[-1].get("role") == "assistant"
                        and not history[-1].get("metadata", {}).get("title")):
                    # 首条 token 到达时，这里原地覆盖掉上面的占位提示
                    history[-1]["content"] = full_response
                else:
                    history.append({"role": "assistant", "content": full_response})
                yield history
            continue

        # ---- updates 流：数据为 {节点名: 该节点返回的更新字典} ----
        if mode == "updates":
            for _node, update in payload.items():
                if not isinstance(update, dict):
                    continue

                # 工具节点（search_context / my_search）产出的 ToolMessage
                for message in update.get("messages", []):
                    if isinstance(message, ToolMessage):
                        # 工具提示已经说明在干什么了，占位提示就该退场
                        if (history and isinstance(history[-1], dict)
                                and history[-1].get("content") == _PLACEHOLDER):
                            history.pop()
                        title = ("🛠️ 工具调用: 互联网搜索" if message.name == "my_search"
                                 else f"🛠️ 工具调用: {message.name}")
                        history.append(ChatMessage(
                            role="assistant",
                            content=f"🔧 已调用工具 `{message.name}`：\n{str(message.content)[:300]}",
                            metadata={"title": title},
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
        # 写入响应到 Milvus（把本轮最终结果保存到历史对话上下文库）
        # 必须 await：若丢进后台任务而不等待，任务可能随本次请求结束被取消，写库会静默丢失
        await save_final_answer(current_state.values)


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
    # 先预热再开界面，避免把权重加载算进用户第一句话的等待时间
    warm_up_embedding()

    # 启动 Gradio 应用
    instance.launch(
        debug=True,
        theme=gr.themes.Soft(),
        css=css
    )
