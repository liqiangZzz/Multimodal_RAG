from typing import Any, List

from langchain_core.embeddings import Embeddings
from ragas.embeddings.base import BaseRagasEmbedding
from sentence_transformers import SentenceTransformer


# ============================================================
# gme-Qwen2-VL 多模态嵌入（LangChain + Ragas 双接口，自包含）
# ============================================================
class ModernQwen2Embeddings(Embeddings, BaseRagasEmbedding):
    """gme-Qwen2-VL-2B-Instruct 嵌入，同时适配两套接口：

    - LangChain `Embeddings` 接口（embed_query / embed_documents）：
      供 SemanticChunker 等组件直接使用；
    - Ragas `BaseRagasEmbedding` 接口（embed_text / embed_texts 及异步版本）：
      供 RAG 评估链路使用。

    设计要点：
      - 模型带自定义模块，必须传 trust_remote_code=True；
      - 输入必须是 dict 形式（{"text": ...} 或 {"image": ...}）；
      - 输出可能是 torch.Tensor / numpy.ndarray，统一转 list。
    """

    def __init__(
            self,
            model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            *,
            local_files_only: bool = True,
            device: str = None,
    ):
        self.model_name = model_name
        self._model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
            device=device,  # None 时由 sentence-transformers 自动选择
        )

    # ---------------- 内部：统一的编码逻辑 ----------------
    def _encode(self, inputs: List[dict]) -> List[List[float]]:
        """执行实际编码，返回 List[List[float]]。"""
        embeddings = self._model.encode(
            inputs,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        # numpy array 直接 .tolist()
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        # torch tensor 逐条转换
        return [
            e.cpu().tolist() if hasattr(e, "cpu") else list(e)
            for e in embeddings
        ]

    def _to_text_inputs(self, texts: List[str]) -> List[dict]:
        """文本 → gme 要求的 dict 输入格式。"""
        return [{"text": t} for t in texts]

    # ---------------- LangChain 同步接口 ----------------
    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本（LangChain 接口）。"""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入一批文本（LangChain 接口）。"""
        return self._encode(self._to_text_inputs(texts))

    # ---------------- Ragas 同步接口 ----------------
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """同步嵌入单条文本。"""
        return self._encode(self._to_text_inputs([text]))[0]

    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """同步嵌入多条文本。"""
        return self._encode(self._to_text_inputs(texts))

    # ---------------- Ragas 异步接口 ----------------
    # 底层 sentence-transformers 是同步的，无法真正异步，
    # 直接同步调用即可（MPS/CPU 场景下多线程收益有限）。
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """异步嵌入单条文本（底层同步）。"""
        return self.embed_text(text)

    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """异步嵌入多条文本（底层同步）。"""
        return self.embed_texts(texts)
