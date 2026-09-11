from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class CustomQwen3Embeddings(Embeddings):
    """自定义一个qwen3的Embedding和langchain整合的类"""


    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name, local_files_only=True)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts)