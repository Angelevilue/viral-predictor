# 模型兼容性与错误处理

本文档详细说明了Viral Predictor应用程序中的模型兼容性和错误处理机制。

## 概述

Viral Predictor支持多种LLM提供商和模型，但不同模型具有不同的能力和限制。特别是在处理JSON输出格式和返回值类型方面存在差异。本文档解释了我们如何处理这些差异，确保应用程序能够与各种模型无缝协作。

## 模型能力识别

### 不支持JSON输出的模型

在`config/model_config.py`中，我们维护了一个不支持JSON输出格式的模型列表：

```python
NON_JSON_FORMAT_MODELS = [
    "deepseek-reasoner",  # DeepSeek的推理模型不支持JSON输出
    "deepseek-coder",     # DeepSeek的代码模型可能也不支持
    "hunyuan"             # 腾讯混元模型可能不支持
]
```

同时提供了一个辅助函数来检查模型是否支持JSON输出：

```python
def supports_json_format(model: str) -> bool:
    """检查模型是否支持JSON输出格式"""
    return model not in NON_JSON_FORMAT_MODELS
```

## API调用策略

在`llms/llm.py`的`predict_engagement`方法中，我们根据模型的能力采用不同的API调用策略：

### 支持JSON输出的模型

对于支持JSON输出的模型（如OpenAI的GPT系列），我们使用`response_format`参数：

```python
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
```

### 不支持JSON输出的模型

对于不支持JSON输出的模型（如DeepSeek Reasoner），我们：

1. 在提示词中明确要求JSON格式输出：

```python
prompt = f"{prompt}\n\n请以JSON格式返回结果，格式如下：\n{{\"like\": 数字, \"comment\": 数字, \"share\": 数字, \"quote\": 数字}}\n请确保返回的是有效的JSON格式，不要添加额外的文本。数值必须是整数，不要使用布尔值。"
```

2. 不使用`response_format`参数：

```python
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
```

## 响应处理

### JSON解析

我们首先尝试标准的JSON解析：

```python
json_data = json.loads(prediction)
```

### 数据类型处理

对于成功解析的JSON，我们处理各种可能的数据类型：

```python
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
```

### 错误恢复

如果JSON解析失败，我们使用正则表达式提取数值或布尔值：

```python
# 提取like值
like_match = re.search(r'"like"\s*:\s*(true|false|\d+)', prediction, re.IGNORECASE)
like = 1 if like_match and like_match.group(1).lower() == "true" else 0
if like_match and like_match.group(1).isdigit():
    like = int(like_match.group(1))
```

## 错误日志

为了便于调试，我们记录详细的错误信息：

```python
logger.error(f"JSON解析错误: {str(e)}, 内容: {prediction}")
```

## 默认值

在任何情况下，我们都确保返回有效的结果对象：

```python
return {"like": 0, "comment": 0, "share": 0, "quote": 0}
```

## 最佳实践

1. **添加新模型时**：
   - 测试模型是否支持JSON输出格式
   - 如果不支持，将其添加到`NON_JSON_FORMAT_MODELS`列表中

2. **调试API调用问题**：
   - 检查日志中的错误信息
   - 验证API密钥和基础URL是否正确
   - 确认模型名称拼写正确

3. **优化提示词**：
   - 对于不支持JSON输出的模型，可能需要调整提示词以获得更一致的输出格式

## 未来改进

1. 添加更多模型的支持和测试
2. 实现更智能的JSON提取算法
3. 添加模型响应缓存以提高性能
4. 实现自动模型能力检测
