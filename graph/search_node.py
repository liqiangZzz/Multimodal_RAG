import asyncio
from typing import Dict

from langchain_core.messages import ToolMessage

from embedding.gme_qwen2_vl_2b_embedding import call_local_model
from graph.custom_state import MultimodalRAGState
from graph.graph_db.collections_operator_graph import CONTEXT_COLLECTION_NAME, client
from graph.graph_db.db_retriever_graph import MilvusRetriever
from utils.log_utils import log

retriever = MilvusRetriever(
    collection_name=CONTEXT_COLLECTION_NAME,
    milvus_client=client
)

#  自定义是为了替代：由LangGraph框架自带的ToolNode（有大模型动态传参 来调用工具）
class SearchContextToolNode:
    """自定义类，来执行搜索上下文工具"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}


    async def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []

        # 并行执行所有工具调用
        tasks = []
        for tool_call in message.tool_calls:
            if tool_call.get("args") and 'query' in tool_call["args"]:
                query = tool_call["args"]["query"]
                log.info(f"开始从上下文中检索：{query}")
            else:
                query = inputs.get('input_text')

            # 使用异步调用（工具参数名是 username，状态里的字段也是 username）
            task = self.tools_by_name[tool_call["name"]].ainvoke(
                {'query': query, 'username': inputs.get('username')}
            )
            tasks.append((tool_call, task))

        # 等待所有异步调用完成
        tool_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (tool_call, _), tool_result in zip(tasks, tool_results):
            if isinstance(tool_result, Exception):
                # 错误处理
                tool_result = f"工具执行错误: {str(tool_result)}"

            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

def _extract_entity(hit) -> Dict:
    """从 Milvus 检索结果中统一取出实体字段。

    - RRF 融合结果: {"id":.., "score":.., "entity": {...}}
    - dense_search 返回值: pymilvus Hit 对象（带 .entity 属性）
    """
    if isinstance(hit, dict):
        return hit.get("entity", hit)
    return getattr(hit, "entity", hit)


def retriever_node(state: MultimodalRAGState):
    """检索历史对话上下文并返回"""
    # 与 tools.search_context 保持一致的 username 过滤：
    # 之前这里不过滤用户，会把**其他用户**的历史对话检索出来直接拼进 third_chatbot 的提示词
    # （多用户下的隐私泄漏；日志里能看到无过滤的命中）。
    username = state.get('username')
    filter_expr = f'username == "{username}"' if username else None

    if state.get('input_type') == 'only_image':
        log.info(f"开始从历史对话上下文中检索图片：{state.get('input_image')}")
        # 构建图像输入数据
        input_data = [{'image': state.get('input_image')}]
        # 调用API获取图像嵌入向量
        ok, embedding, status, retry_after = call_local_model(input_data)
        #  密入向量检索：图像嵌入向量
        # 注意：dense_search(embedding, ...) 参数顺序，向量在前
        results = retriever.dense_search(embedding, limit=3, filter_expr=filter_expr) if ok else []
    else:
        input_text = (state.get('input_text') or "").strip()
        # 构建文本输入数据
        input_data = [{'text': input_text}]
        # 调用API获取嵌入向量
        ok, embedding, status, retry_after = call_local_model(input_data)

        #  混合检索：文本 + 嵌入向量
        # 注意：hybrid_search(input_text, embedding, ...) 参数顺序，文本在前、向量在后
        results = retriever.hybrid_search(
            input_text, embedding, limit=3, filter_expr=filter_expr
        ) if ok else []
    log.info(f"从历史对话上下文中检索到的结果：{results}")

    # 返回文档内容
    docs = []
    for hit in results:
        entity = _extract_entity(hit)
        docs.append({
            "context_text": entity.get("context_text"),
            "username": entity.get("username"),
            "timestamp": entity.get("timestamp"),
            "message_type": entity.get("message_type"),
        })

    # 返回检索结果
    log.info(f"返回检索结果：{docs}")
    return {'context_retrieved': docs, 'image_retrieved': []}