from pymilvus import MilvusClient, DataType, Function, FunctionType

from utils.env_utils import MILVUS_URI

# 长期历史记录存储的集合名称
CONTEXT_COLLECTION_NAME = 't_context_collection'

# context_text 字段的上限（字符数）。schema 定义与写入端截断必须引用同一个常量，
# 改这里即可同时生效；超限内容由 save_context 在入库前截断，否则服务端会整条拒绝。
MAX_CONTEXT_TEXT_LENGTH = 6000

# ========== 1. 连接 Milvus ==========
client = MilvusClient(
    uri=MILVUS_URI,
    collection_name=CONTEXT_COLLECTION_NAME
    # user="root",        # 若开启了认证
    # password="Milvus",
)


# ========== 2. 创建 Collection ==========
def create_store_collection():
    #  先删除已存在的同名 Collection（开发阶段）
    if client.has_collection(CONTEXT_COLLECTION_NAME):
        client.drop_collection(CONTEXT_COLLECTION_NAME)

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 某一条聊天记录的文本内容
    schema.add_field(field_name="context_text", datatype=DataType.VARCHAR, max_length=MAX_CONTEXT_TEXT_LENGTH,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})

    # 用户名
    schema.add_field(field_name="username", datatype=DataType.VARCHAR, max_length=100, nullable=True)
    schema.add_field(field_name="timestamp", datatype=DataType.INT64, nullable=True)
    # 消息类型
    schema.add_field(field_name="message_type", datatype=DataType.VARCHAR, max_length=100, nullable=True)
    # 归一化后的提问，作为写库幂等判重的等值键（save_context.normalize_question）
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2000, nullable=True)

    #  稀疏向量字段
    schema.add_field(field_name="context_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    #  密集向量字段
    schema.add_field(field_name="context_dense", datatype=DataType.FLOAT_VECTOR, dim=1536)

    # 文本BM25函数
    bm25_function = Function(
        name="text_bm25_emb",  # 文本BM25函数
        input_field_names=["context_text"],  # 输入字段
        output_field_names=["context_sparse"],  # 输出字段
        function_type=FunctionType.BM25,  # 函数类型
    )

    # 添加BM25函数到集合
    schema.add_function(bm25_function)
    # 准备索引参数
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="context_sparse",
        index_name="context_sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # 稀疏向量倒排索引
        metric_type="BM25",  # 指标类型： BM25相似度 （用于文本检索）倒排索引
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,  # 1.2 ~ 2.0 (1.2) 词频 (TF) 的饱和度: 高频词的贡献越大，词频影响越线性，饱和度增长越慢(通俗：控制一个词出现多少次才算“多”)
            "bm25_b": 0.75
            # 0.0 ~ 1.0 (0.75) 文档长度归一化的强度： 文档长度的影响越大，对长文档的惩罚越强（通俗：控制“长篇大论”相对于“言简意赅”的劣势有多大，旨在避免长文档仅仅因为包含更多词汇而在相似度计算中占据不公平的优势。）
        },
    )

    index_params.add_index(
        field_name="context_dense",
        index_name="context_dense_inverted_index",
        index_type="AUTOINDEX",  # 自动索引
        metric_type="IP",  # 指标类型： IP相似度
    )

    client.create_collection(
        collection_name=CONTEXT_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


if __name__ == '__main__':
    create_store_collection()
    # 查看集合信息
    res = client.describe_collection(
        collection_name=CONTEXT_COLLECTION_NAME
    )

    print(res)
