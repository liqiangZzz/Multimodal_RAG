from typing import Literal, Optional, Dict, List

from langgraph.graph import MessagesState


class MultimodalRAGState(MessagesState):
    """
    多模态RAG的状态类
    """

    input_type: Literal["has_text", "only_image"]  # 用户输入类型
    context_retrieved: Optional[List[Dict[str, str]]]  # 从向量数据库中检索到的文本内容
    image_retrieved: Optional[List[str]]  # 从向量数据库中检索到的图像URL 路径
    web_search_result: Optional[str] = None  # 网络搜索结果（由 web_search_node 写入）

    needs_retrieval: Optional[bool] = False  # 是否需要重新检索
    evaluate_score: Optional[float] = None  # 评估指标的分数
    final_response: Optional[str] = None  # 最终的响应

    input_image: Optional[str] = None  # 用户输入的图像 ，里面是base64编码的图片
    input_text: Optional[str] = None  # 用户输入的文本

    username: str = "ZS"  # 用户名

    human_answer: Optional[str] = "rejected"  # 用户是否同意 RAG 的最终响应
