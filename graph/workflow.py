"""多轮对话 RAG 工作流的命令行入口。

节点函数与图结构全部在 graph.graph_builder，本文件只保留命令行输入输出外壳，
与 workflow_gradio.py 共用同一份图，两者不会再有流程差异。

运行（必须在项目根以模块方式运行，直接跑脚本会因项目根不在 sys.path 而报
No module named 'graph'）：

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m graph.workflow
"""

import asyncio
import os

from langchain_core.messages import HumanMessage

from embedding.gme_qwen2_vl_2b_embedding import image_to_base64
from graph.graph_builder import graph, new_run_config, save_final_answer, update_state, warm_up_embedding
from graph.print_messages import pretty_print_messages

config = new_run_config()


async def execute_graph(user_input: str) -> str:
    """执行一次工作流（新提问或回复人工审批）"""
    result = ''  # AI 助手的最后一条消息

    current_state = graph.get_state(config)  # 得到实时的状态（短期上下文）
    if current_state.next:  # 出现了工作流的中断 -> 本次输入是人工审批答复
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

        message = HumanMessage(content=[])
        if text:
            message.content.append({"type": "text", "text": text})
        if image_base64:
            message.content.append(image_base64)

        async for chunk in graph.astream({'messages': [message]}, config, stream_mode='values'):
            pretty_print_messages(chunk, last_message=True)

    # 收尾处理：两条路径（新输入 / 中断恢复）统一在这里判断，
    # 不能在中断分支里提前 return，否则会跳过下面的写库逻辑。
    current_state = graph.get_state(config)
    if current_state.next:  # 仍在中断点（等待人工审批）
        result = ("由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n "
                  "如果认可，请输入“approve”，否则请输入“rejected”，系统将会重新生成回复！")
    else:
        # 写入响应到 Milvus（把本轮最终结果保存到历史对话上下文库）
        await save_final_answer(current_state.values)

    return result


async def main():
    while True:
        user_input = input('用户输入(文本和图片用&隔开)：')
        if user_input.lower() in ['exit', 'quit', '退出']:
            break

        res = await execute_graph(user_input)
        if res:
            print('AI: ', res)


if __name__ == '__main__':
    # 先预热再开界面，避免把权重加载算进用户第一句话的等待时间
    warm_up_embedding()

    asyncio.run(main())
