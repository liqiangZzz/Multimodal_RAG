"""
通用 HuggingFace 模型下载脚本（国内网络适配）
"""
import os
import time

# =====================================================================
# 1. 配置区 (想下载别的模型，修改这里即可)
# =====================================================================
# 模型在 HuggingFace 上的仓库 ID
REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"

# 用于判断缓存是否命中的标志性文件
KEY_FILE = "config.json"

# 国内镜像源（huggingface.co 直连超时，改用镜像）
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# 最大重试次数
MAX_RETRIES = 3


def setup_mirror():
    """设置 HF 镜像源环境变量。"""
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = DEFAULT_ENDPOINT
    # 关键：关闭 Xet 传输协议。
    # HF 新版默认用 Xet 分片下载，会去 cas-server.xethub.hf.co 取数据，
    # 走镜像时该域名返回 401 导致下载失败，必须降级为传统 HTTP 下载。
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # 弱网下放宽超时，避免大文件下载中断
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    print(f"[配置] HF_ENDPOINT        = {os.environ['HF_ENDPOINT']}")
    print(f"[配置] HF_HUB_DISABLE_XET = {os.environ['HF_HUB_DISABLE_XET']}")


def is_cached(repo_id: str, filename: str):
    """
    检查文件是否已在本地缓存中。

    返回缓存中的实际路径，未命中返回 None。
    用 huggingface_hub 官方 API 而非自己拼路径，避免版本差异导致路径算错。
    """
    from huggingface_hub import try_to_load_from_cache

    result = try_to_load_from_cache(repo_id, filename)
    # 命中时返回 str 路径；未命中返回 None（部分版本返回 _CACHED_NO_EXIST 哨兵对象）
    return result if isinstance(result, str) else None


def main():
    setup_mirror()

    from huggingface_hub import snapshot_download

    # =================================================================
    # 2. 缓存命中则跳过
    # =================================================================
    cached = is_cached(REPO_ID, KEY_FILE)
    if cached:
        print(f"[跳过] 模型已在缓存中：{cached}")
        print("       如需强制重新下载，先执行：")
        print(f"       rm -rf ~/.cache/huggingface/hub/models--{REPO_ID.replace('/', '--')}")
        return cached

    print(f"[开始] 下载 {REPO_ID}")
    print("       首次下载约 1.2GB，请耐心等待...")

    # =================================================================
    # 3. 下载到 HuggingFace 标准缓存目录
    #    循环下载，最多重试 3 次
    # =================================================================
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 下载到 HuggingFace 标准缓存目录
            cache_dir = snapshot_download(
                repo_id=REPO_ID,
                # 只下载推理必需的文件，跳过原始 ckpt 以节省空间/时间
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.py", "*.md"],
                # 排除大模型原始权重格式
                ignore_patterns=["*.bin", "original/*", "*.msgpack", "*.h5", "*.ot"],
            )
            print(f"[完成] 模型已缓存到：{cache_dir}")
            return cache_dir

        except Exception as e:
            print(f"[警告] 第 {attempt} 次下载失败，原因：{e}")
            if attempt == MAX_RETRIES:
                print("[错误] 已达到最大重试次数，请检查网络后重新运行。")
                raise
            print(f"       5 秒后进行第 {attempt + 1} 次重试...")
            time.sleep(5)


if __name__ == "__main__":
    main()
