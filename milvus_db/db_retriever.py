"""
Milvus 检索模块。

提供以下检索方式：
    - dense_search         : 密集向量检索（语义相似）
    - sparse_search        : 稀疏检索 / BM25 全文检索
    - image_search         : 以图搜图 / 以图搜文
    - multimodal_search    : 图文融合查询
    - hybrid_search        : 客户端 RRF 融合
    - hybrid_search_native : 服务端原生 RRF 融合（Milvus 2.4+）

所有可调参数通过 RetrieverConfig 统一管理，调用时可按需覆盖。

pymilvus Hit 对象访问方式：
    hit.id           -> 主键
    hit.distance     -> 距离 / 分数
    hit.entity       -> entity 字典
    hit["text"]      -> 等价于 hit.entity["text"]
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from pymilvus import MilvusClient

from milvus_db.collections_operator import MILVUS_COLLECTION_NAME
from utils.env_utils import MILVUS_URI
from utils.gme_qwen2_vl_2b_embedding import get_image_embedding, get_fused_embedding, get_text_embedding


# ============================================================
# 配置
# ============================================================

@dataclass
class RetrieverConfig:
    """Milvus 检索器的全部可调参数。

    所有字段都有合理默认值，需要时在构造 RetrieverConfig 时覆盖即可。
    """

    # 字段名
    dense_field: str = "dense"
    sparse_field: str = "sparse"

    # 默认返回字段
    output_fields: List[str] = field(default_factory=lambda: [
        "text", "category", "filename", "image_path", "title",
    ])

    # 密集检索参数
    dense_metric_type: str = "IP"
    dense_nprobe: int = 10

    # 稀疏检索参数
    sparse_metric_type: str = "BM25"
    sparse_drop_ratio: float = 0.2  # 0-1 之间，用于控制 BM25 检索结果的随机性

    # 混合检索参数
    rrf_k: int = 60
    candidate_multiplier: int = 2  # 每路取 limit * multiplier 个候选
    both_hit_bonus: float = 1.2  # 两路都命中的加权系数，1.0 表示不加权

    # 默认返回条数
    top_k: int = 5


# ============================================================
# 检索器
# ============================================================

class MilvusRetriever:
    """Milvus 检索器。

    所有检索方法都接受 limit / filter_expr / output_fields 等可选参数，
    未传时使用 self.config 中的默认值。
    """

    def __init__(
            self,
            collection_name: str,
            milvus_client: MilvusClient,
            config: Optional[RetrieverConfig] = None):
        """
        Args:
            collection_name: Milvus 集合名称
            milvus_client:   MilvusClient 实例
            config:          可调配置，不传则使用 RetrieverConfig 默认值
        """
        self.collection_name = collection_name
        self.milvus_client = milvus_client
        self.config = config or RetrieverConfig()

    # ============================================================
    # 内部统一搜索入口
    # ============================================================
    def _search(
            self,
            data: List[Any],
            anns_field: str,
            metric_type: str,
            params: Dict,
            limit: int,
            output_fields: Optional[List[str]] = None,
            filter_expr: Optional[str] = None,
    ) -> List:
        """统一的底层搜索方法。

        所有对外方法最终都调用它，避免重复拼参数。
        filter_expr 为空时不会传 filter 参数。

        Args:
            data:           输入数据，每个元素为一个向量
            anns_field:     向量字段名
            metric_type:    指标类型
            params:         指标类型参数
            limit:          返回条数
            output_fields:  返回字段列表，默认 self.config.output_fields
            filter_expr:    过滤表达式(可选)

        Returns:
            搜索结果列表，每个元素为一个 Hit 对象
        """
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": data,
            "anns_field": anns_field,
            "limit": limit,
            "output_fields": output_fields or self.config.output_fields,
            "search_params": {"metric_type": metric_type, "params": params},
        }

        if filter_expr:
            kwargs["filter"] = filter_expr

        # 分数越大越相似，所以返回的列表是有序的
        return self.milvus_client.search(**kwargs)[0]

    # ============================================================
    # 密集检索
    # ============================================================
    def dense_search(
            self,
            query_embedding: List[float],
            limit: Optional[int] = None,
            filter_expr: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
    ) -> List:
        """密集向量检索。

        Args:
            query_embedding: 已向量化好的查询向量（1536 维）
            limit:           返回结果数量，默认 self.config.top_k
            filter_expr:     可选过滤表达式，如 "category == 'image'"
            output_fields:   可选返回字段，默认 self.config.output_fields

        Returns:
            Milvus Hit 对象列表
        """
        return self._search(
            data=[query_embedding],
            anns_field=self.config.dense_field,
            metric_type=self.config.dense_metric_type,
            params={"nprobe": self.config.dense_nprobe},
            limit=limit or self.config.top_k,
            output_fields=output_fields,
            filter_expr=filter_expr,
        )

    # ============================================================
    # 稀疏检索
    # ============================================================
    def sparse_search(
            self,
            query_text: str,
            limit: Optional[int] = None,
            filter_expr: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
    ) -> List:
        """稀疏向量检索。

        Args:
            query_text:         关键词文本（原始字符串，Milvus 内部用 BM25 自动编码）
            limit:         返回结果数量，默认 self.config.top_k
            filter_expr:   可选过滤表达式，如 "category == 'image'"
            output_fields: 可选返回字段，默认 self.config.output_fields

        Returns:
            Milvus Hit 对象列表
        """
        return self._search(
            data=[query_text],
            anns_field=self.config.sparse_field,
            metric_type=self.config.sparse_metric_type,
            params={"drop_ratio_search": self.config.sparse_drop_ratio},
            limit=limit or self.config.top_k,
            output_fields=output_fields,
            filter_expr=filter_expr,
        )

    # ============================================================
    # 以图搜图 / 以图搜文
    # ============================================================
    def image_search(
            self,
            image_path: str,
            limit: Optional[int] = None,
            only_image: bool = False,
            filter_expr: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
    ) -> List:
        """以图搜图 / 以图搜文。

        Args:
            image_path:    本地图片路径（或 URL）
            limit:         返回结果数量
            only_image:    True 则只搜 category == 'image'
            filter_expr:   自定义过滤表达式（优先级高于 only_image）
            output_fields: 可选返回字段

        Returns:
            Milvus Hit 对象列表
        """
        query_embedding = get_image_embedding(image_path)
        if not query_embedding:
            print(f"[警告] 图片无效，无法生成向量：{image_path}")
            return []

        # filter_expr 优先，其次按 only_image 生成
        if filter_expr and only_image:
            print("[警告] 同时传了 filter_expr 和 only_image=True，将优先使用 filter_expr")
        final_filter = filter_expr or ("category == 'image'" if only_image else None)

        # 密集检索
        return self.dense_search(
            query_embedding=query_embedding,
            limit=limit,
            filter_expr=final_filter,
            output_fields=output_fields,
        )

    # ============================================================
    # 图文融合检索
    # ============================================================
    def multimodal_search(
            self,
            text: str,
            image_path: str,
            limit: Optional[int] = None,
            only_image: bool = False,
            filter_expr: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
    ) -> List:
        """图文融合查询：文本 + 图片一起编码成一个向量再搜。

        Args:
            text:          文本查询
            image_path:    图片路径（或 URL）
            limit:         返回结果数量
            only_image:    True 则只搜 category == 'image'，
            filter_expr:   自定义过滤表达式（优先级高于 only_image）
            output_fields: 可选返回字段

        Returns:
            Milvus Hit 对象列表
        """
        query_embedding = get_fused_embedding(text, image_path)
        if not query_embedding:
            print(f"[警告] 图文融合查询无效，无法生成向量：{text}，{image_path}")
            return []

        if filter_expr and only_image:
            print("[警告] 同时传了 filter_expr 和 only_image=True，将优先使用 filter_expr")
        final_filter = filter_expr or ("category == 'image'" if only_image else None)

        return self.dense_search(
            query_embedding=query_embedding,
            limit=limit,
            filter_expr=final_filter,
            output_fields=output_fields,
        )

    # ============================================================
    # 混合检索
    # ============================================================

    def _rrf_fuse(
            self,
            dense_hits: List,
            sparse_hits: List,
            limit: int,
            rrf_k: Optional[int] = None,
    ) -> List[Dict]:
        """RRF 融合两路结果。

        RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
        两路都命中的文档会乘以 both_hit_bonus（默认 1.2）。
        """
        rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k

        rrf_scores: Dict[int, float] = {}
        id_to_hit: Dict[int, Any] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            id_to_hit[hit.id] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            id_to_hit.setdefault(hit.id, hit)

        # 两路都命中的加权
        if self.config.both_hit_bonus != 1.0:
            hit_count: Counter = Counter()
            for hit in dense_hits:
                hit_count[hit.id] += 1
            for hit in sparse_hits:
                hit_count[hit.id] += 1
            for doc_id in rrf_scores:
                if hit_count[doc_id] >= 2:
                    rrf_scores[doc_id] *= self.config.both_hit_bonus

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        return [
            {
                "id": doc_id,
                "score": rrf_scores[doc_id],
                "entity": id_to_hit[doc_id].entity,
            }
            for doc_id in sorted_ids[:limit]
        ]

    def hybrid_search(
            self,
            query_text: str,
            query_embedding: List[float],
            limit: Optional[int] = None,
            rrf_k: Optional[int] = None,
            filter_expr: Optional[str] = None,
    ) -> List[Dict]:
        """混合检索：客户端 RRF 融合。

        Args:
            query_text:      用于稀疏检索的关键词文本
            query_embedding: 用于密集检索的向量
            limit:           最终返回的结果数
            rrf_k:           RRF 常数，默认 self.config.rrf_k
            filter_expr:     可选过滤表达式

        Returns:
            融合排序后的结果列表，每条包含 id、score、entity
        """
        limit = limit or self.config.top_k
        candidate_limit = limit * self.config.candidate_multiplier

        dense_hits = self.dense_search(
            query_embedding,
            limit=candidate_limit,
            filter_expr=filter_expr,
        )
        sparse_hits = self.sparse_search(
            query_text,
            limit=candidate_limit,
            filter_expr=filter_expr,
        )

        return self._rrf_fuse(dense_hits, sparse_hits, limit, rrf_k)

    # ============================================================
    # 原生混合检索
    # ============================================================
    def hybrid_search_native(
            self,
            query_text: str,
            query_embedding: List[float],
            limit: Optional[int] = None,
            rrf_k: Optional[int] = None,
            output_fields: Optional[List[str]] = None
    ) -> List:
        """原生混合检索（Milvus 2.4+ 支持）。

        在服务端一次性完成两路检索和 RRF 融合，性能更好。

        Args:
            query_text:      用于稀疏检索的关键词文本
            query_embedding: 用于密集检索的向量
            limit:           最终返回的结果数
            rrf_k:           RRF 常数
            output_fields:   可选返回字段

        Returns:
            Milvus Hit 对象列表
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        limit = limit or self.config.top_k
        candidate_limit = limit * self.config.candidate_multiplier

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field=self.config.dense_field,
            param={
                "metric_type": self.config.dense_metric_type,
                "params": {"nprobe": self.config.dense_nprobe},
            },
            limit=candidate_limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=self.config.sparse_field,
            param={
                "metric_type": self.config.sparse_metric_type,
                "params": {"drop_ratio_search": self.config.sparse_drop_ratio},
            },
            limit=candidate_limit,
        )
        res = self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k or self.config.rrf_k),
            limit=limit,
            output_fields=output_fields or self.config.output_fields,
        )
        return res[0]


# ============================================================
# 测试入口
# ============================================================

def _print_hits(title: str, hits: List, show_category: bool = True) -> None:
    """统一打印检索结果，减少 main 里的重复代码。
    Args:
        title:       打印标题
        hits:        检索结果列表
        show_category: 是否显示分类，默认 True
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] distance={hit.distance:.4f}, id={hit.id}")
        if show_category:
            print(f"    category={hit.entity.get('category')}")
            print(f"    image_path={hit.entity.get('image_path') or 'N/A'}")
        print(f"    title: {hit.entity.get('title', '')}")
        print(f"    text:  {hit.entity.get('text', '')[:80]}...")
    print("-----------------" * 6)


def _print_fused_hits(title: str, results: List[Dict]) -> None:
    """统一打印 RRF 融合结果。

    Args:
        title:   打印标题
        results: 融合后的结果列表
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] RRF score={r['score']:.6f}, id={r['id']}")
        print(f"    category={r['entity'].get('category')}")
        print(f"    image_path={r['entity'].get('image_path') or 'N/A'}")
        print(f"    title: {r['entity'].get('title', '')}")
        print(f"    text:  {r['entity'].get('text', '')[:80]}...")
    print("-----------------" * 6)


if __name__ == '__main__':
    # 1. 初始化 Milvus 客户端
    client = MilvusClient(uri=MILVUS_URI)

    # 2. 自定义配置初始化检索器（也可以不传 config），默认使用默认配置。
    config = RetrieverConfig(
        top_k=5,
        dense_nprobe=10,
        sparse_drop_ratio=0.2,
        rrf_k=60,
        both_hit_bonus=1.2
    )

    retriever = MilvusRetriever(
        collection_name=MILVUS_COLLECTION_NAME,
        milvus_client=client,
        config=config,
    )

    # ============================================================
    # 文本检索
    # ============================================================
    query = "Flink 是什么？"
    query_embedding = get_text_embedding(query)

    # ===========密集检索=============
    _print_hits("密集检索（语义相似）",
                retriever.dense_search(query_embedding, limit=3))

    # ===========稀疏检索=============
    _print_hits("稀疏检索（BM25 关键词）",
                retriever.sparse_search(query, limit=3))

    # ===========混合检索=============
    _print_fused_hits(
        "混合检索（RRF 融合）",
        retriever.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            limit=5,
        ),
    )

    # ============================================================
    # 图片检索
    # ============================================================
    test_image = "/Users/Python/project/project-learn/python-code/Multimodal_RAG/data/flink_stateful_stream_checkpoint_architecture.png"

    # ===========以图搜图（只搜图片）=============
    _print_hits("以图搜图（only_image=True）",
                retriever.image_search(test_image, limit=3, only_image=True))

    # =========== 以图搜文（文本 + 图片混合）===========
    _print_hits(
        "以图搜文（混合结果）",
        retriever.image_search(test_image, limit=3, only_image=False),
    )

    # ============================================================
    # 图文融合（only_image=False）查看混合结果
    # ============================================================
    _print_hits(
        "图文融合（only_image=False）",
        retriever.multimodal_search(
            text="Flink 是什么",
            image_path=test_image,
            limit=10,
            only_image=False,
        ),
    )
