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

## Requirements

- Python 3.x
- Streamlit
- OpenAI API access (via OpenRouter)
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get an API key from OpenRouter
4. Run the application: `streamlit run viral_predictor.py`

## How It Works

1. Enter two versions of your content
2. Select your target platform
3. Set the number of simulated users
4. Enter your OpenRouter API key
5. Click "Predict" to see how users might engage with your content

The app simulates user behavior and provides statistical confidence scores for engagement metrics, helping you choose the most effective version of your content.

## License

This project is licensed under the MIT License.
