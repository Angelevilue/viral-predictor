import streamlit as st
from openai import OpenAI
import json
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

st.title("Viral Predictor")
st.write("##### Simulate how users react to your content so you know it'll go viral before you post")
def calc_confidence(users, vote_a, vote_b):
    if vote_a == 0 and vote_b == 0:
        return "-", 0.0
    
    if vote_a == 0:
        return "B", 100.0
    if vote_b == 0:
        return "A", 100.0
    
    try:
        if vote_a >= vote_b:
            z_stat, vote_confidence_a = proportions_ztest(
                count=[vote_a, vote_b],
                nobs=[users, users],
                alternative='larger'
            )
            vote_confidence_a = (1 - vote_confidence_a) * 100
            return "A", vote_confidence_a if not np.isnan(vote_confidence_a) else 50.0
        else:
            z_stat, vote_confidence_b = proportions_ztest(
                count=[vote_b, vote_a],
                nobs=[users, users],
                alternative='larger'
            )
            vote_confidence_b = (1 - vote_confidence_b) * 100
            return "B", vote_confidence_b if not np.isnan(vote_confidence_b) else 50.0
    except:
        total_votes = vote_a + vote_b
        if vote_a > vote_b:
            return "A", (vote_a / total_votes) * 100
        else:
            return "B", (vote_b / total_votes) * 100

input_a, input_b = st.columns(2)
input_a.write("### Version A")
version_a = input_a.text_area("Enter your content here", key="version_a", value = "", height=200)
input_b.write("### Version B")
version_b = input_b.text_area("Enter your content here", key="version_b", value = "", height=200)
platform = input_a.selectbox("Platform", ["Twitter", "TikTok", "Instagram", "LinkedIn", "Facebook", "Hacker News", "Reddit", "Blog Post"])
max_users = input_b.number_input("Max Users", value=10)
api_key = input_a.text_input("OpenRouter API Key", value="") # TODO: remove this
model = input_b.text_input("Model", value="openai/gpt-4o")
predict_button = st.button("Predict")
chart_empty = st.empty()
chart_data = {
    "engagement_a": [0],
    "engagement_b": [0],
    "users": [0]
}
chart_empty.line_chart(chart_data, x="users", y=["engagement_a", "engagement_b"], )

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

c = st.container()
like_col, comment_col, share_col, quote_col = c.columns(4)
like_col.write("### Likes")
like_empty = like_col.empty()
like_empty.write("0% Vers. ?")
comment_col.write("### Comments")
comment_empty = comment_col.empty()
comment_empty.write("0% Vers. ?")
share_col.write("### Shares")
share_empty = share_col.empty()
share_empty.write("0% Vers. ?")
quote_col.write("### Quotes")
quote_empty = quote_col.empty()
quote_empty.write("0% Vers. ?")

if predict_button:
    # Initialize counters
    users = 0

    like_a = 0
    comment_a = 0
    share_a = 0
    quote_a = 0
    total_a = 0

    like_b = 0
    comment_b = 0
    share_b = 0
    quote_b = 0
    total_b = 0

    prompt_a = f"""Imagine you are a random user on {platform}. You came across the following content:
    '''
    {version_a}
    '''
    Decide whether to like, comment, share (retweet, repost, etc.), quote, or not.
    Output your decision as a JSON object with the following fields:
    - like: bool
    - comment: bool
    - share: bool
    - quote: bool"""

    prompt_b = f"""Imagine you are a random user on {platform}. You came across the following content:
    '''
    {version_b}
    '''
    Decide whether to like, comment, share (retweet, repost, etc.), quote, or not.
    Output your decision as a JSON object with the following fields:
    - like: bool
    - comment: bool
    - share: bool
    - quote: bool"""


    while users < max_users:
        users += 1

        completion_a = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": prompt_a
                }
            ],
            response_format={"type": "json_object"}
        )
        prediction_a = completion_a.choices[0].message.content
        prediction_a = json.loads(prediction_a)
        like_a += prediction_a["like"]
        comment_a += prediction_a["comment"]
        share_a += prediction_a["share"]
        quote_a += prediction_a["quote"]
        total_a = like_a + comment_a + share_a + quote_a

        completion_b = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": prompt_b
                }
            ],
            response_format={"type": "json_object"}
        )
        prediction_b = completion_b.choices[0].message.content
        prediction_b = json.loads(prediction_b)
        like_b += prediction_b["like"]
        comment_b += prediction_b["comment"]
        share_b += prediction_b["share"]
        quote_b += prediction_b["quote"]
        total_b = like_b + comment_b + share_b + quote_b

        chart_data["engagement_a"].append(total_a)
        chart_data["engagement_b"].append(total_b)
        chart_data["users"].append(users)
        chart_empty.line_chart(chart_data, x="users", y=["engagement_a", "engagement_b"])

        # Calculate statistical confidence
        like_winner, like_confidence = calc_confidence(users, like_a, like_b)
        like_empty.write(f"{like_confidence:.2f}% Vers. {like_winner}")

        comment_winner, comment_confidence = calc_confidence(users, comment_a, comment_b)
        comment_empty.write(f"{comment_confidence:.2f}% Vers. {comment_winner}")

        share_winner, share_confidence = calc_confidence(users, share_a, share_b)
        share_empty.write(f"{share_confidence:.2f}% Vers. {share_winner}")

        quote_winner, quote_confidence = calc_confidence(users, quote_a, quote_b)
        quote_empty.write(f"{quote_confidence:.2f}% Vers. {quote_winner}")
