import streamlit as st
from openai import OpenAI
import json
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import asyncio
from openai import AsyncOpenAI

st.set_page_config(layout="wide")
st.title("Viral Predictor")
st.write("##### Simulate how users react to your content so you know it'll go viral before you post")
# set page to wide
video_col = st.columns([1.5,2.5,1.5])[1]
video_col.write("#### Example usage")
video_col.video("example.mp4")
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

chart_data = {
    "engagement_a": [0],
    "engagement_b": [0],
    "users": [0]
}
standard_batch_size = 5
input_a, input_b = st.columns(2)
input_a.write("### Version A")
version_a = input_a.text_area("Enter your content here", key="version_a", value = "", height=200)
input_b.write("### Version B")
version_b = input_b.text_area("Enter your content here", key="version_b", value = "", height=200 )
platform = input_a.selectbox("Platform", ["Twitter", "TikTok", "Instagram", "LinkedIn", "Facebook", "Hacker News", "Reddit", "Blog Post"])
max_users = input_b.number_input("Max Users", value=20, step=10)
api_key = input_a.text_input("OpenRouter API Key", value="") # TODO: remove this
model = input_b.text_input("Model", value="openai/gpt-4o")
predict_button = st.button("Predict")

st.write("### Cumulative Engagement")
chart_empty = st.empty()
chart_empty.line_chart(chart_data, x="users", y=["engagement_a", "engagement_b"])

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

st.write("### Statistical Confidence")
c = st.container()
like_col, comment_col, share_col, quote_col = c.columns(4)
like_col.write("##### Likes")
like_empty = like_col.empty()
like_empty.write("0% Vers. ?")
comment_col.write("##### Comments")
comment_empty = comment_col.empty()
comment_empty.write("0% Vers. ?")
share_col.write("##### Shares")
share_empty = share_col.empty()
share_empty.write("0% Vers. ?")
quote_col.write("##### Quotes")
quote_empty = quote_col.empty()
quote_empty.write("0% Vers. ?")

async def get_prediction(prompt, model):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            if completion and hasattr(completion, 'choices') and completion.choices and len(completion.choices) > 0:
                prediction = completion.choices[0].message.content
                return json.loads(prediction)
            else:
                # If we get an empty response, log it and retry
                print(f"Empty response received on attempt {attempt+1}, retrying...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Return a default response if all retries fail
                    print("All retries failed, returning default response")
                    return {"like": False, "comment": False, "share": False, "quote": False}
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                # Return a default response if all retries fail
                print("All retries failed, returning default response")
                return {"like": False, "comment": False, "share": False, "quote": False}

async def main():
    if predict_button:
        # Initialize counters
        users = 0
        like_a = comment_a = share_a = quote_a = total_a = 0
        like_b = comment_b = share_b = quote_b = total_b = 0

        prompt_a = f"""You are scrolling through {platform} and came across the following content:
        '''
        {version_a}
        '''
        Decide whether to like, comment, share (retweet, repost, etc.), or quote.
        Output your decision as a JSON object with the following fields:
        - like: bool
        - comment: bool
        - share: bool
        - quote: bool"""

        prompt_b = f"""You are scrolling through {platform} and came across the following content:
        '''
        {version_b}
        '''
        Decide whether to like, comment, share (retweet, repost, etc.), or quote.
        Output your decision as a JSON object with the following fields:
        - like: bool
        - comment: bool
        - share: bool
        - quote: bool"""

        while users < max_users:
            # 每次预测5个用户（或剩余的用户数）
            batch_size = min(standard_batch_size, max_users - users)
            predictions_a = []
            predictions_b = []
            
            # 并行获取多个用户的预测
            tasks_a = [get_prediction(prompt_a, model) for _ in range(batch_size)]
            tasks_b = [get_prediction(prompt_b, model) for _ in range(batch_size)]
            
            predictions_a = await asyncio.gather(*tasks_a)
            predictions_b = await asyncio.gather(*tasks_b)
            
            # 处理这批预测结果
            for pred_a, pred_b in zip(predictions_a, predictions_b):
                users += 1
                
                like_a += pred_a["like"]
                comment_a += pred_a["comment"]
                share_a += pred_a["share"]
                quote_a += pred_a["quote"]
                total_a = like_a + comment_a + share_a + quote_a

                like_b += pred_b["like"]
                comment_b += pred_b["comment"]
                share_b += pred_b["share"]
                quote_b += pred_b["quote"]
                total_b = like_b + comment_b + share_b + quote_b

                chart_data["engagement_a"].append(total_a)
                chart_data["engagement_b"].append(total_b)
                chart_data["users"].append(users)

            # Update the chart
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

if __name__ == "__main__":
    asyncio.run(main())
