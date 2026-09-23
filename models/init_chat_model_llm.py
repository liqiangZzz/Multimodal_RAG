"""创建项目共享的聊天模型与外部服务客户端实例。

业务代码统一从这里导入，不要在各自模块里重复构造客户端 ——
重复实例化既浪费连接资源，也会让超时、重试这类参数散落各处而失去统一控制。

新增或替换供应商时，请在此处一并补上对应的 client 及其超时 / 重试配置。
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from openai import OpenAI, AsyncOpenAI
from zhipuai import ZhipuAI

from utils.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GLM_API_KEY, GLM_BASE_URL, ZHIPU_API_KEY

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

glm_llm_flash_client = OpenAI(
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
    # 显式超时：不配置时 SDK 用自己的默认值（约 600s），一旦接口挂起，
    # 单次请求就能拖住十几分钟。换服务商后请按其响应特征重新评估。
    timeout=60.0,
    max_retries=1,
)

async_glm_llm_flash_client = AsyncOpenAI(
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
    # 异步客户端供 ragas 评委等链路复用。不显式配置时 SDK 自带默认超时并自动重试，
    # 遇到服务端不响应会把整条工作流一起拖住（见 graph/tools.py 的 CONTEXT_EVAL_TIMEOUT）。
    # 这里收紧到 60s + 1 次重试，让失败尽快暴露给上层的超时逻辑。
    timeout=60.0,
    max_retries=1,
)

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)