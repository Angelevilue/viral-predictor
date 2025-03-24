# Viral Predictor

A Streamlit application that helps content creators simulate user engagement before posting by leveraging AI to predict how users might react to different versions of content.

## Features

- A/B test two versions of your content
- Supports multiple platforms (Twitter, TikTok, Instagram, LinkedIn, Facebook, Hacker News, Reddit, Blog Posts)
- Real-time engagement predictions for:
  - Likes
  - Comments
  - Shares
  - Quotes
- Statistical confidence scoring
- Live engagement visualization
- Multi-language support (English/Chinese)
- Multiple LLM providers support:
  - OpenAI
  - OpenRouter
  - SiliconFlow
  - Nebius
  - Aliyun
  - Zhipuai
  - DeepSeek
  - Tencent
- Error handling for API calls and model compatibility

## Requirements

- Python 3.x
- Streamlit
- API keys for your preferred LLM providers
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your API keys in the `.env` file:

```
# OpenAI API key
OPENAI_API_KEY=your_openai_api_key

# DeepSeek API key
DeepSeek_API_KEY=your_deepseek_api_key

# Tencent API key
TENCENT_API_KEY=your_tencent_api_key

# Add other API keys as needed
```

4. Run the application using the startup script:

```bash
python run_app.py
```

Or with custom options:

```bash
python run_app.py --port 8502 --language zh
```

## How It Works

1. Enter two versions of your content
2. Select your target platform
3. Choose your preferred LLM provider and model
4. Set the number of simulated users
5. Click "Predict" to see how users might engage with your content

The app simulates user behavior and provides statistical confidence scores for engagement metrics, helping you choose the most effective version of your content.

## Advanced Features

### Multi-Language Support

The application supports both English and Chinese interfaces. You can switch languages using the language toggle button in the top-right corner or by specifying the language when starting the application.

### Multiple LLM Providers

The application supports various LLM providers with different capabilities. The system automatically detects model capabilities and adjusts API calls accordingly, ensuring compatibility with models that:
- Support JSON output format
- Do not support JSON output format
- Return boolean values instead of numbers

### Error Handling

The application includes error handling for API calls and model compatibility issues. If an error occurs, the application will display an error message and provide guidance on how to resolve the issue.

### Custom Startup Options

The `run_app.py` script provides several customization options:
- `--port`: Specify the port for the Streamlit server (default: 8501)
- `--language`: Set the default language (en/zh, default: en)
- `--debug`: Enable debug mode

## License

This project is licensed under the MIT License.
