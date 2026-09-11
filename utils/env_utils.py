"""
环境变量工具模块

该模块用于加载和管理项目所需的环境变量配置。
使用 python-dotenv 从 .env 文件中读取环境变量，
并提供 API 相关的配置常量。
"""
import os

from dotenv import load_dotenv

# =====================================================================
# 1. 加载 .env —— 系统环境变量优先，文件只补齐缺失项
# =====================================================================
# 加载 .env 文件中的环境变量。
# override=False 表示：PyCharm / 系统环境变量优先，.env 只补充缺失变量。
# 这样 .env 可以使用 ${MYSQL_PASSWORD} 这类外部变量拼接配置，
# 又不会用 .env 里的占位符覆盖 PyCharm 传入的真实值。
load_dotenv(override=False)


# =====================================================================
# 2. 导出模型配置
# =====================================================================
# API 密钥
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
GLM_API_KEY = os.getenv('GLM_API_KEY')

# URL
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
GLM_BASE_URL = os.getenv('GLM_BASE_URL')

