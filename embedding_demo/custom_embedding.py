from typing import Any, List

from langchain_core.embeddings import Embeddings
from ragas.embeddings.base import BaseRagasEmbedding
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Qwen3 文本嵌入（LangChain 接口）
# ============================================================
class CustomQwen3Embeddings(Embeddings):
    """自定义一个 qwen3 文本嵌入，适配 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.qwen3_embedding = SentenceTransformer(model_name, local_files_only=True)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.qwen3_embedding.encode(texts)


# ============================================================
# 2. gme-Qwen2-VL 多模态嵌入（直接实现 Ragas 接口，自包含）
# ============================================================
class ModernQwen2Embeddings(BaseRagasEmbedding):
    """gme-Qwen2-VL-2B-Instruct 嵌入，实现 Ragas BaseRagasEmbedding 接口。

    设计要点：
      - 对外提供同步 / 异步的 embed_text / embed_texts 方法；
      - 内部 _encode 统一处理 dict 输入和 tensor→list 的转换；

    注意：
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
