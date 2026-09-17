"""创建项目共享的 DeepSeek 聊天模型实例。

普通示例统一导入本模块的 ``deepseek_llm``。只有专门演示模型初始化方式或
需要特殊模型配置时，才在对应示例中单独调用 ``init_chat_model``。
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from utils.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GLM_API_KEY, GLM_BASE_URL

# =====================================================================
# 1. 创建共享模型 —— 供项目内普通示例统一复用
# =====================================================================

deepseek_llm_pro: BaseChatModel = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    extra_body={
        "thinking": {"type": "disabled"}  # 关闭思考模式
    }
)

deepseek_llm_flash: BaseChatModel = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    # api_base 是 ChatDeepSeek 的原生服务地址字段。
    api_base=DEEPSEEK_BASE_URL,
    # 关闭思考模式，使基础示例的响应更直接，并保持与原公共模型配置一致。
    extra_body={"thinking": {"type": "disabled"}},
)

glm_llm_flash: BaseChatModel = init_chat_model(
    model="glm-5.3-flash",
    model_provider="openai",
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)
