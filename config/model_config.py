"""
模型配置文件
包含各个LLM提供商的配置信息
"""
import os
from typing import Dict, Any, List

# 模型配置
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "openrouter": {
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "models": ["openai/gpt-4o", "anthropic/claude-3-opus", "anthropic/claude-3-sonnet"]
    },
    "siliconflow": {
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "models": ["Pro/deepseek-ai/DeepSeek-R1", "Pro/deepseek-ai/DeepSeek-V3"]
    },
    "nebius": {
        "base_url": os.getenv("Nebius_BASE_URL", "https://api.studio.nebius.ai/v1"),
        "api_key": os.getenv("Nebius_DeepSeek_API_KEY", ""),
        "models": ["deepseek-ai/DeepSeek-V3"]
    },
    "aliyun": {
        "base_url": os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("ALIYUN_API_KEY", ""),
        "models": ["qwen-max-latest", "deepseek-v3", "deepseek-r1"]
    },
    "zhipuai": {
        "base_url": os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "api_key": os.getenv("ZHIPUAI_API_KEY", ""),
        "models": ["glm-4-plus", "glm-4"]
    },
    "deepseek": {
        "base_url": os.getenv("DeepSeek_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DeepSeek_API_KEY", ""),
        "models": ["deepseek-reasoner", "deepseek-coder"]
    },
    "tencent": {
        "base_url": os.getenv("TENCENT_BASE_URL", "https://api.lkeap.cloud.tencent.com/v1"),
        "api_key": os.getenv("TENCENT_API_KEY", ""),
        "models": ["deepseek-r1", "hunyuan"]
    }
}

# 不支持JSON输出格式的模型列表
NON_JSON_FORMAT_MODELS = [
    "deepseek-reasoner",  # DeepSeek的推理模型不支持JSON输出
    "deepseek-coder",     # DeepSeek的代码模型可能也不支持
    "hunyuan"             # 腾讯混元模型可能不支持
]

def get_available_providers() -> List[str]:
    """获取所有可用的模型提供商"""
    return list(MODEL_CONFIGS.keys())

def get_available_models(provider: str) -> List[str]:
    """获取指定提供商的可用模型"""
    if provider in MODEL_CONFIGS:
        return MODEL_CONFIGS[provider]["models"]
    return []

def get_provider_config(provider: str) -> Dict[str, Any]:
    """获取指定提供商的配置"""
    if provider in MODEL_CONFIGS:
        return MODEL_CONFIGS[provider]
    raise ValueError(f"不支持的提供商: {provider}。支持的提供商: {list(MODEL_CONFIGS.keys())}")

def supports_json_format(model: str) -> bool:
    """检查模型是否支持JSON输出格式"""
    return model not in NON_JSON_FORMAT_MODELS
