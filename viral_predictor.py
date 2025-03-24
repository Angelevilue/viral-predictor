import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# 导入自定义模块
from llms.llm import ViralPredictionLLM
from prompt.content_prediction import get_engagement_prompt
from config.language import TEXTS

# 初始化会话状态
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # 默认英文

# 语言切换函数
def toggle_language():
    """切换语言"""
    if st.session_state.language == 'en':
        st.session_state.language = 'zh'
    else:
        st.session_state.language = 'en'
    st.rerun()  # 重新运行应用以应用新语言

# 获取当前语言的文本
def get_text(key):
    """获取当前语言的文本"""
    return TEXTS[st.session_state.language][key]

# 转义特殊字符
def escape_markdown(text):
    """转义Markdown中的特殊字符"""
    if not isinstance(text, str):
        return text
    return st.markdown(text, escape=True)

# 设置页面配置
st.set_page_config(layout="wide", page_title=get_text("title"))

# 设置页面样式
st.markdown("""
<style>
    .main-header {text-align: center; margin-bottom: 10px;}
    .sub-header {text-align: center; margin-bottom: 30px;}
    .stButton>button {width: 100%;}
    .result-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .metric-container {
        padding: 15px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .metric-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .language-toggle {
        position: absolute;
        top: 10px;
        right: 20px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    h4 {
        margin-bottom: 0.3rem;
    }
    .row-widget.stSelectbox, .row-widget.stNumberInput {
        padding-top: 0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 添加自定义CSS样式
st.markdown("""
<style>
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    /* 自定义语言按钮样式 */
    div.lang-button {
        position: absolute;
        top: 1rem;
        right: 1rem;
        z-index: 999;
    }
    div.lang-button button {
        background-color: #ff5252;
        color: white;
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        border: none;
    }
    /* 标题容器样式 */
    .title-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .title-text {
        flex-grow: 1;
        text-align: center;
    }
    .lang-switch-container {
        width: 80px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    h4 {
        margin-bottom: 0.3rem;
    }
    .row-widget.stSelectbox, .row-widget.stNumberInput {
        padding-top: 0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 创建标题和语言切换按钮布局
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown(f'<div class="title-text"><h1 class="main-header">{get_text("title")}</h1></div>', unsafe_allow_html=True)

# 添加语言切换按钮（放在标题右侧）
st.markdown('<div class="lang-switch-container">', unsafe_allow_html=True)
current_lang = "中文" if st.session_state.language == "en" else "EN"
if st.button(f'🌐 {current_lang}', key='lang_toggle'):
    toggle_language()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(f'<h4 class="sub-header">{get_text("subtitle")}</h4>', unsafe_allow_html=True)

# 创建输入表单
st.markdown("---")
st.markdown(f"### {get_text('input_section') if 'input_section' in TEXTS[st.session_state.language] else '内容设置'}")

# 版本A和版本B的输入区域
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"#### {get_text('version_a')}", help=None)
    version_a = st.text_area(
        label=get_text('version_a'),
        placeholder=get_text('input_placeholder'),
        height=200, 
        key="version_a_input", 
        label_visibility="collapsed"
    )

with col_b:
    st.markdown(f"#### {get_text('version_b')}", help=None)
    version_b = st.text_area(
        label=get_text('version_b'),
        placeholder=get_text('input_placeholder'),
        height=200, 
        key="version_b_input", 
        label_visibility="collapsed"
    )

# 平台和模型选择
col_platform, col_provider = st.columns(2)

with col_platform:
    st.markdown(f"#### {get_text('platform')}", help=None)
    platform = st.selectbox(
        label=get_text('platform'),
        options=["Twitter", "Facebook", "Instagram", "LinkedIn", "TikTok"], 
        key="platform_select", 
        label_visibility="collapsed"
    )

with col_provider:
    st.markdown(f"#### {get_text('model_provider')}", help=None)
    provider = st.selectbox(
        label=get_text('model_provider'),
        options=ViralPredictionLLM.get_available_providers(), 
        key="provider_select", 
        label_visibility="collapsed",
        index=ViralPredictionLLM.get_available_providers().index("tencent") if "tencent" in ViralPredictionLLM.get_available_providers() else 0
    )

# 模型选择和最大用户数
col_model, col_users = st.columns(2)

with col_model:
    st.markdown(f"#### {get_text('model')}", help=None)
    available_models = ViralPredictionLLM.get_available_models(provider)
    model = st.selectbox(
        label=get_text('model'),
        options=available_models, 
        key="model_select", 
        label_visibility="collapsed",
        index=available_models.index("deepseek-r1") if "deepseek-r1" in available_models else 0
    )

with col_users:
    st.markdown(f"#### {get_text('max_users')}", help=None)
    max_users = st.number_input(
        label=get_text('max_users'),
        min_value=10, 
        max_value=100, 
        value=20, 
        step=1, 
        key="max_users_input", 
        label_visibility="collapsed"
    )
    standard_batch_size = 5  # 每批处理的用户数

# 预测按钮
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
with predict_col2:
    predict_button = st.button(get_text("predict"), use_container_width=True)

# 添加间隙
st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# 图表数据
chart_data = {
    "engagement_a": [0],
    "engagement_b": [0],
    "users": [0]
}

async def get_prediction(prompt, llm_client):
    """使用LLM客户端获取预测结果"""
    return await llm_client.predict_engagement(prompt)

async def main():
    if predict_button:
        # 初始化计数器
        users = 0
        like_a = comment_a = share_a = quote_a = total_a = 0
        like_b = comment_b = share_b = quote_b = total_b = 0

        # 初始化LLM客户端
        try:
            llm_client = ViralPredictionLLM(provider=provider, model=model)
            st.info(get_text("using_model").format(provider, model))
        except Exception as e:
            st.error(get_text("init_model_failed").format(str(e)))
            return

        # 获取当前语言的提示词模板
        prompt_template = get_engagement_prompt(st.session_state.language)
        
        # 创建提示词
        prompt_a = prompt_template.format(platform=platform, content=version_a)
        prompt_b = prompt_template.format(platform=platform, content=version_b)

        progress_bar = st.progress(0)
        
        while users < max_users:
            # 每次预测5个用户（或剩余的用户数）
            batch_size = min(standard_batch_size, max_users - users)
            predictions_a = []
            predictions_b = []
            
            # 并行获取多个用户的预测
            tasks_a = [get_prediction(prompt_a, llm_client) for _ in range(batch_size)]
            tasks_b = [get_prediction(prompt_b, llm_client) for _ in range(batch_size)]
            
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
                
                # 更新图表数据
                chart_data["engagement_a"].append(total_a)
                chart_data["engagement_b"].append(total_b)
                chart_data["users"].append(users)
                
                # 更新进度条
                progress_bar.progress(users / max_users)
        
        # 显示结果
        st.success(get_text("prediction_complete").format(users))
        st.markdown("---")
        
        # 显示累计互动图表
        st.subheader(get_text('cumulative_engagement'))
        
        chart_df = {
            get_text("users"): chart_data["users"][1:],
            f"{get_text('version')} A": chart_data["engagement_a"][1:],
            f"{get_text('version')} B": chart_data["engagement_b"][1:]
        }
        
        st.line_chart(chart_df, x=get_text("users"))
        
        # 显示统计置信度
        st.subheader(get_text('statistical_confidence'))
        
        # 创建结果卡片 - 使用st.container()替代HTML div
        with st.container():
            # 使用expander组件代替自定义HTML
            with st.expander(get_text("total_engagement"), expanded=True):
                # 显示总互动量结果
                winner, confidence = calc_confidence(users, total_a, total_b)
                if confidence > 0:
                    if winner == "A":
                        st.success(f"{get_text('better_version').format(winner, confidence)}")
                    else:
                        st.info(f"{get_text('better_version').format(winner, confidence)}")
                else:
                    st.write(f"{get_text('no_difference')}")
        
        # 显示详细指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.expander(get_text("likes"), expanded=True):
                st.write(f"A: {like_a} | B: {like_b}")
                winner, confidence = calc_confidence(users, like_a, like_b)
                if confidence > 0:
                    st.write(f"{get_text('better_version').format(winner, confidence)}")
                else:
                    st.write(f"{get_text('no_difference')}")
        
        with col2:
            with st.expander(get_text("comments"), expanded=True):
                st.write(f"A: {comment_a} | B: {comment_b}")
                winner, confidence = calc_confidence(users, comment_a, comment_b)
                if confidence > 0:
                    st.write(f"{get_text('better_version').format(winner, confidence)}")
                else:
                    st.write(f"{get_text('no_difference')}")
        
        with col3:
            with st.expander(get_text("shares"), expanded=True):
                st.write(f"A: {share_a} | B: {share_b}")
                winner, confidence = calc_confidence(users, share_a, share_b)
                if confidence > 0:
                    st.write(f"{get_text('better_version').format(winner, confidence)}")
                else:
                    st.write(f"{get_text('no_difference')}")
        
        with col4:
            with st.expander(get_text("quotes"), expanded=True):
                st.write(f"A: {quote_a} | B: {quote_b}")
                winner, confidence = calc_confidence(users, quote_a, quote_b)
                if confidence > 0:
                    st.write(f"{get_text('better_version').format(winner, confidence)}")
                else:
                    st.write(f"{get_text('no_difference')}")

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
        return "-", 0.0

if __name__ == "__main__":
    # 运行主应用程序
    asyncio.run(main())
