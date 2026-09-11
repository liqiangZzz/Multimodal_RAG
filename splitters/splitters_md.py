import base64
import hashlib
import io
import os
import re
from typing import List, Tuple

from PIL import Image
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter

from embedding_demo.custom_embedding import CustomQwen3Embeddings
from utils.common_utils import get_sorted_md_files
from utils.log_utils import log


class MarkdownDirSplitter:
    # 类变量：统一管理图片匹配正则，避免硬编码
    # 说明：!\[.*?\] 匹配 ![alt](...) 中的 alt 部分（允许空或非空），比 !\[\] 更健壮
    # 能同时匹配 jpg、jpeg、png 等所有 data:image 开头的 Base64 图片。
    IMAGE_PATTERN = re.compile(r'!\[.*?\]\(data:image/(.*?);base64,(.*?)\)', re.DOTALL)

    def __init__(self, images_output_dir: str, text_chunk_size: int = 1000):
        """
        初始化分割器
        Args:
            images_output_dir: md文件中的图片存放目录
            text_chunk_size：文本块大小（超过这个大小会触发语义分割）
        """

        self.images_output_dir = images_output_dir
        self.text_chunk_size = text_chunk_size
        os.makedirs(self.images_output_dir, exist_ok=True)  # 确保图片存储目录存在

        # 1. 结构分割器：根据 Markdown 的标题层级进行初步切分
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        self.text_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)  # 文本分割器

        # 2. 语义分割器：当文本块太长时，根据语义（Embedding相似度）进行智能切分，避免硬切
        self.embedding = CustomQwen3Embeddings("Qwen/Qwen3-Embedding-0.6B")  # 语义嵌入
        self.semantic_splitter = SemanticChunker(
            self.embedding, breakpoint_threshold_type="percentile"
        )

    def save_base64_to_image(self, base64_str: str, output_path: str) -> None:
        """
        将 base64 字符串解码为图片并保存到本地。
        Args:
            base64_str: base64字符串
            output_path: 输出路径
        """

        # 去掉 base64 的前缀 (例如 "data:image/png;base64,")
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(',', 1)[1]

        img_data = base64.b64decode(base64_str)  # base64 解码为二进制数据
        img = Image.open(io.BytesIO(img_data))  # 将二进制转为 PIL 图片对象

        # 确保图片以 RGB 模式保存，防止因为透明通道（RGBA）导致保存为 JPEG 时报错
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        img.save(output_path)  # 保存图片到指定路径

    def extract_and_replace_images(self, content: str, source: str) -> Tuple[List[Document], str]:
        """
        提取 Markdown 中的 Base64 图片，并将原文本中的乱码替换为 [图片] 占位符。
        注意：字符串是不可变的，本函数不会修改原始 content，而是生成并返回一份新的纯文本

        Args:
            content: 包含 Base64 乱码的 Markdown 原始内容
            source: 当前 Md 文件的路径
        Returns:
            Tuple[List[Document], str]: 一个包含两个元素的元组 (image_docs, cleaned_content)

            1. image_docs (List[Document]):
               提取出的图片 Document 列表。
               - page_content 是图片保存到本地的【绝对路径】（如 /Users/.../xxx.png）。
               - metadata 标记了 'embedding_type': 'image'。
               - 用途：后续专门用于【图片向量化（多模态）】入库。

            2. cleaned_content (str):
               清理后的【全新】纯文本字符串。
               - 原本那一大坨极其消耗 Token 的 "data:image/png;base64,..." 乱码，
                 被全部替换为了 '[图片]' 占位符。
               - 用途：后续专门用于【纯文本切分与向量化】。
               - 为什么要留占位符？为了防止上下文断裂（例如“如上图所示”），
                 同时避免大模型在计算 Embedding 时超出 Token 限制而崩溃。
        """
        image_docs = []

        def replace_image(match):
            """
            正则替换的回调函数。
            re.sub 每匹配到一处 Base64 图片，就会调用一次这个函数。
            """
            img_type = match.group(1).split('/')[-1]
            base64_data = match.group(2)

            # 使用 MD5 生成唯一的文件名，防止重名覆盖
            hash_key = hashlib.md5(base64_data.encode()).hexdigest()
            filename = f"{hash_key}.{img_type if img_type in ['png', 'jpg', 'jpeg'] else 'png'}"
            image_path = os.path.join(self.images_output_dir, filename)  # 生成图片保存路径

            # 1. 保存图片
            self.save_base64_to_image(base64_data, image_path)

            # 2. 为图片创建一个单独的 Document，用于后续的图片 Embedding（多模态）做准备
            image_docs.append(Document(
                page_content=str(image_path),
                metadata={
                    "source": source,
                    "alt_txt": "图片",
                    "embedding_type": "image"  # 标记这是图片类型
                }
            ))

            return "[图片]"  # 3. 将替换结果返回给 re.sub，用于拼接新文本

        # content 的值，如下
        # Flink 简介
        # Flink 是一个分布式处理引擎。
        # ![图片](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==)
        # 如上图所示，Flink 的性能非常强大。
        #         ⬇️
        # 经过 re.sub 会遍历 content，将匹配到的 Base64 全部替换为 [图片]，并生成新字符串 cleaned_content
        # ✅      🟰
        # Flink 是一个分布式处理引擎。
        # [图片]
        # 如上图所示，Flink 的性能非常强大。
        cleaned_content = self.IMAGE_PATTERN.sub(replace_image, content)

        # 返回两个完全独立的结果：图片列表 和 清理后的新文本
        return image_docs, cleaned_content

    def process_md_file(self, md_file: str) -> List[Document]:
        """
        处理单个 Md 文件，返回 Document 列表（包含内部标题层级修复）
        Args:
            md_file: md文件路径，如：第一章 Apache Flink 概述/第一章 Apache Flink 概述_page_0.md
        Returns:
            Document列表
        """

        with open(md_file, "r", encoding="utf-8") as file:
            content = file.read()

        # 1. 先按标题结构（#、##、###）切分
        split_documents: List[Document] = self.text_splitter.split_text(content)

        documents = []
        for doc in split_documents:
            # 2. 判断当前切块中是否包含 Base64 图片
            if self.IMAGE_PATTERN.search(doc.page_content):

                # 提取图片，并直接获得替换后的纯文本（元组解包）
                image_docs, cleaned_content = self.extract_and_replace_images(doc.page_content, md_file)

                # ⚠️【核心修复】去掉 [图片] 占位符后，判断是否还有实质性的文字内容
                # 如果是纯图片段落，去掉占位符后就是空的，不需要存入文本库
                text_without_images = cleaned_content.replace("[图片]", "").strip()

                if text_without_images:
                    doc.metadata['embedding_type'] = 'text'
                    documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))

                # 把图片 Document 也加入列表
                documents.extend(image_docs)
            else:
                doc.metadata['embedding_type'] = 'text'
                documents.append(doc)

        # 3. 语义二次切分
        final_docs = []
        for doc in documents:
            if len(doc.page_content) > self.text_chunk_size:
                # 如果当前文档块超过阈值，则进行语义切分
                final_docs.extend(self.semantic_splitter.split_documents([doc]))
            else:
                final_docs.append(doc)

        # 4. 补充标题层级信息（在单文件层面处理，防止跨文件的标题互相污染）
        return self.add_title_hierarchy(final_docs, md_file)

    def add_title_hierarchy(self, documents: List[Document], source_filename: str) -> List[Document]:
        """
        为切分后的文档补充完整的标题层级结构。
        例如：某一段只属于 "### 3.1 步骤"，它本身没有 H1 和 H2。
        这个函数会把上文中的 H1 和 H2 补充到这条数据的 metadata 里。
        """
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []

        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata['source'] = source_filename

            # 更新标题状态
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key in new_metadata:
                    current_titles[level] = new_metadata[header_key]
                    # 如果有了 H2，就把 H3 清空
                    for lower_level in range(level + 1, 4):
                        current_titles[lower_level] = ""

            # 为缺失标题的层级填充上当前的标题
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key not in new_metadata:
                    new_metadata[header_key] = current_titles[level]
                elif current_titles[level] != new_metadata[header_key]:
                    new_metadata[header_key] = current_titles[level]

            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
            )

        return processed_docs

    def process_md_dir(self, md_dir: str, source_filename: str) -> List[Document]:
        """
        遍历整个目录，处理所有 Md 文件，返回合并后的 Document 列表
        Args:
            md_dir: md文件目录
            source_filename: 源文件名
        Returns:
            Document列表
        """

        md_files = get_sorted_md_files(md_dir)  # 获取md文件列表
        print(f"md文件列表为： {md_files}")

        documents = []
        for md_file in md_files:
            log.info(f"真正处理的文件为： {md_file}")
            documents.extend(self.process_md_file(md_file))

        # 这里不再重复调用 add_title_hierarchy。
        # 因为 process_md_file 内部已经处理好了每个独立文件的标题层级。
        # 目录层级的操作只需统一将 source 替换为传入的外部 PDF 文件名，避免重复计算和状态污染。
        for doc in documents:
            doc.metadata['source'] = source_filename

        return documents


if __name__ == '__main__':
    md_dir = r'/Users/Python/project/project-learn/python-code/Multimodal_RAG/output/第一章 Apache Flink 概述'

    splitter = MarkdownDirSplitter(images_output_dir=r'/Users/Python/project/project-learn/python-code/Multimodal_RAG/images')
    docs = splitter.process_md_dir(md_dir, source_filename='第一章 Apache Flink 概述.pdf')

    # 打印结果
    for i, doc in enumerate(docs):
        print(f"\n文档 #{i + 1}:")
        print(doc)
        print(f"内容: {doc.page_content[:30]}...")
        print(f"元数据: {doc.metadata}...")

        print(f"一级标题: {doc.metadata.get('Header 1', '')}")
        print(f"二级标题: {doc.metadata.get('Header 2', '')}")
        print(f"三级标题: {doc.metadata.get('Header 3', '')}")