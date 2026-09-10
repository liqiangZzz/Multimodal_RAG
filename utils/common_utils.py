import os.path
import re
import shutil
from pathlib import Path
from typing import List, Union


def get_filename(file_path, with_extension=True):
    """
    获取文件名
    Args:
        file_path: 文件绝对路径
        with_extension: 是否包含扩展名
    Returns:
        str: 如果 with_extension 为 True，返回带后缀的文件名（如 test.pdf）；
             如果为 False，返回不带后缀的文件名（如 test）。
    """
    if with_extension:
        return os.path.basename(file_path)
    else:
        return Path(file_path).stem


def get_sorted_md_files(input_dir: str) -> List[str]:
    """
    按照页号，把所有的md文件排序。（xx_0.md, xx_1.md, xx_2.md, .... xx_12.md）
    获取指定目录下所有 .md 文件，并按照 _page_X 中的 X 数值排序
    Args:
        input_dir: 输入目录的路径
    Returns:
        List[str]: 排序后的 md 文件列表（按页码从小到大排序）
    """

    # 1. 获取目录下所有 .md 文件
    md_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.md')]

    # 2. 定义排序 key 函数，提取 _page_ 后的数字
    # 该函数返回页码数字，若没有匹配到数字，则返回无穷大，排在最后
    def sort_key(file_path: str) -> Union[int, float]:

        # 从完整路径中提取文件名（如 "doc_page_1.md"）
        filename = os.path.basename(file_path)

        # 主匹配：优先查找 _page_数字
        # 例：匹配 "doc_page_12.md" 中的 12
        match = re.search(r'_page_(\d+)', filename)

        # 按优先级依次尝试不同的正则
        patterns = [
            r'_page_(\d+)',  # 格式：xxx_page_1.md
            r'page_(\d+)',  # 格式：xxxpage_1.md 或 page_1.md
            r'_(\d{1,3})\.md$',  # 格式：xxx_1.md (限制1-3位数字)
            r'^(\d+)\.md$'  # 格式：1.md (纯数字)
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))

        # 如果没有找到数字，则返回无穷大（inf），确保这类文件排在所有正常页码文件的后面
        return float('inf')

    # 3. 按照数字排序
    # 使用 sorted() 函数，配合自定义的 sort_key 进行升序排序
    sorted_files = sorted(md_files, key=sort_key)
    return sorted_files


def delete_directory_if_non_empty(dir_path) -> bool:
    """
     删除指定目录（如果该目录存在且非空）
     Args:
        dir_path: 目录路径
    Returns:
        bool: 如果目录存在且成功删除，返回 True；如果目录不存在或为空，返回 False
    """

    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"目录 '{dir_path}' 不存在，无需删除。")
        return False

    # 确认路径是一个目录
    if not os.path.isdir(dir_path):
        print(f"路径 '{dir_path}' 不是一个目录，无需删除。")
        return False

    # 检查目录是否为空
    if not os.listdir(dir_path):
        print(f"目录 '{dir_path}' 为空，无需删除。")
        return False

    # 目录存在且非空，进行删除操作
    try:
        shutil.rmtree(dir_path)
        print(f"成功删除非空目录: '{dir_path}'")
        return True
    except OSError as e:
        print(f"目录 '{dir_path}' 删除失败：{e}")
        return False
