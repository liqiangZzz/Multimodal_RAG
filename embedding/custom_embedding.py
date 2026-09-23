from typing import Any, List

from langchain_core.embeddings import Embeddings
from ragas.embeddings.base import BaseRagasEmbedding

from embedding.gme_qwen2_vl_2b_embedding import encode_inputs


class ModernQwen2Embeddings(Embeddings, BaseRagasEmbedding):
    """gme-Qwen2-VL-2B-Instruct 的 LangChain / Ragas 双接口适配。

    本类只做「协议转换」，不持有也不加载模型：
    权重由 `embedding/gme_qwen2_vl_2b_embedding.py` 以进程内单例持有，
    语义切分 / 文档向量化 / 检索 / RAG 评估全链路共用同一份实例，不会重复加载。

    适配两套接口：
      - LangChain `Embeddings`（embed_query / embed_documents）：供 SemanticChunker 使用；
      - Ragas `BaseRagasEmbedding`（embed_text / embed_texts 及异步版本）：供 RAG 评估链路使用。

    输入为字符串或其列表，输出为 1536 维 L2 归一化向量。
    """

    @staticmethod
    def _to_inputs(texts: List[str]) -> List[dict]:
        """字符串列表 → gme 要求的 dict 输入格式（本类唯一的转换职责）。"""
        return [{"text": t} for t in texts]

    # ---------------- LangChain 接口 ----------------
    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本。"""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入一批文本。"""
        return encode_inputs(self._to_inputs(texts))

    # ---------------- Ragas 接口 ----------------
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """嵌入单条文本。"""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """嵌入多条文本。"""
        return encode_inputs(self._to_inputs(texts))

    # ---------------- Ragas 异步接口 ----------------
    # 底层推理是同步的，无法真正异步，直接转调即可（MPS/CPU 场景多线程收益有限）。
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """异步嵌入单条文本（底层同步）。"""
        return self.embed_text(text)

    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """异步嵌入多条文本（底层同步）。"""
        return self.embed_texts(texts)
