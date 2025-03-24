import os
import json
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import sys
import re

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型配置
from config.model_config import MODEL_CONFIGS, get_available_providers, get_available_models, get_provider_config, supports_json_format

# 配置日志记录器
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

class ViralPredictionLLM:
    """
    统一的LLM接口，用于内容病毒性预测
    """
    def __init__(self, provider: str = "openrouter", model: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            provider: 模型提供商 (openai, openrouter, siliconflow, etc.)
            model: 模型名称，如果为None则使用该提供商的第一个模型
        """
        if provider not in MODEL_CONFIGS:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商: {list(MODEL_CONFIGS.keys())}")
        
        config = get_provider_config(provider)
        
        if not config["api_key"]:
            raise ValueError(f"{provider} API密钥未设置。请在.env文件中设置{provider.upper()}_API_KEY")
        
        # 如果未指定模型，使用该提供商的第一个模型
        if model is None and config["models"]:
            model = config["models"][0]
        elif model not in config["models"]:
            logger.warning(f"模型 {model} 不在 {provider} 的推荐模型列表中")
        
        self.provider = provider
        self.model = model
        self.client = AsyncOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"]
        )
        logger.info(f"初始化 {provider} 客户端，使用模型 {model}")
    
    async def predict_engagement(self, prompt: str) -> Dict[str, Any]:
        """
        使用LLM预测内容的参与度
        
        Args:
            prompt: 提示词
            
        Returns:
            解析后的JSON响应
        """
        try:
            # 检查当前模型是否支持JSON输出格式
            supports_json = supports_json_format(self.model)
            
            # 如果模型不支持JSON输出，在提示词中添加JSON格式要求
            if not supports_json:
                # 添加明确的JSON格式要求
                prompt = f"{prompt}\n\n请以JSON格式返回结果，格式如下：\n{{\"like\": 数字, \"comment\": 数字, \"share\": 数字, \"quote\": 数字}}\n请确保返回的是有效的JSON格式，不要添加额外的文本。数值必须是整数，不要使用布尔值。"
                
                # 创建API调用参数，不包含response_format
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
            else:
                # 对于支持JSON输出的模型，使用response_format参数
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=1024
                )
            
            if completion and hasattr(completion, 'choices') and completion.choices and len(completion.choices) > 0:
                prediction = completion.choices[0].message.content
                
                # 对于不支持JSON输出的模型，尝试从文本中提取JSON
                if not supports_json:
                    # 尝试使用正则表达式提取JSON部分
                    json_match = re.search(r'({[\s\S]*})', prediction)
                    if json_match:
                        prediction = json_match.group(1)
                
                try:
                    # 尝试解析JSON
                    json_data = json.loads(prediction)
                    
                    # 处理可能的布尔值返回
                    result = {}
                    for key in ["like", "comment", "share", "quote"]:
                        if key in json_data:
                            # 如果是布尔值，转换为0或1
                            if isinstance(json_data[key], bool):
                                result[key] = 1 if json_data[key] else 0
                            # 如果是字符串，尝试转换为整数
                            elif isinstance(json_data[key], str):
                                try:
                                    result[key] = int(json_data[key])
                                except ValueError:
                                    # 如果字符串无法转换为整数，检查是否为"true"或"false"
                                    if json_data[key].lower() == "true":
                                        result[key] = 1
                                    elif json_data[key].lower() == "false":
                                        result[key] = 0
                                    else:
                                        result[key] = 0
                            else:
                                # 如果是数字，直接使用
                                result[key] = int(json_data[key])
                        else:
                            # 如果键不存在，设为0
                            result[key] = 0
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}, 内容: {prediction}")
                    # 尝试使用正则表达式提取数值或布尔值
                    try:
                        # 提取like值
                        like_match = re.search(r'"like"\s*:\s*(true|false|\d+)', prediction, re.IGNORECASE)
                        like = 1 if like_match and like_match.group(1).lower() == "true" else 0
                        if like_match and like_match.group(1).isdigit():
                            like = int(like_match.group(1))
                        
                        # 提取comment值
                        comment_match = re.search(r'"comment"\s*:\s*(true|false|\d+)', prediction, re.IGNORECASE)
                        comment = 1 if comment_match and comment_match.group(1).lower() == "true" else 0
                        if comment_match and comment_match.group(1).isdigit():
                            comment = int(comment_match.group(1))
                        
                        # 提取share值
                        share_match = re.search(r'"share"\s*:\s*(true|false|\d+)', prediction, re.IGNORECASE)
                        share = 1 if share_match and share_match.group(1).lower() == "true" else 0
                        if share_match and share_match.group(1).isdigit():
                            share = int(share_match.group(1))
                        
                        # 提取quote值
                        quote_match = re.search(r'"quote"\s*:\s*(true|false|\d+)', prediction, re.IGNORECASE)
                        quote = 1 if quote_match and quote_match.group(1).lower() == "true" else 0
                        if quote_match and quote_match.group(1).isdigit():
                            quote = int(quote_match.group(1))
                        
                        return {"like": like, "comment": comment, "share": share, "quote": quote}
                    except Exception as ex:
                        logger.error(f"正则提取失败: {str(ex)}")
                        # 如果正则提取失败，返回默认值
                        return {"like": 0, "comment": 0, "share": 0, "quote": 0}
            else:
                logger.error("API返回空响应")
                return {"like": 0, "comment": 0, "share": 0, "quote": 0}
                
        except Exception as e:
            logger.error(f"调用API时出错: {str(e)}")
            return {"like": 0, "comment": 0, "share": 0, "quote": 0}
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """获取所有可用的模型提供商"""
        return get_available_providers()
    
    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """获取指定提供商的可用模型"""
        return get_available_models(provider)
