import streamlit as st
import aiohttp
import asyncio
import pandas as pd
from typing import List, Dict
from datetime import datetime

# ========================
# 全局配置
# ========================
DEFAULT_ROLES = ["大學生", "上班族", "游戲主播", "家庭主婦", "中學生", "資深玩家"]
DEFAULT_STYLES = ["幽默吐槽", "專業分析", "情緒激動", "理性討論", "簡短評價"]
DEFAULT_LANGS = ["粵語", "中英夾雜", "網絡用語"]

DEEPSEEK_API_KEY = "sk-c4137b37722c4f74acc3b2ff4dff2fc4"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
SERPAPI_KEY = "6a95d3ee7aaf25c8a1777bf944977a5e4e59765d7d04ada0c9f2f7dc9556246c"
CONCURRENT_REQUESTS = 10

# ========================
# 核心功能函數
# ========================
async def web_search(query: str) -> List[Dict]:
    """執行網絡搜索"""
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google"
            }
            async with session.get("https://serpapi.com/search", params=params) as response:
                data = await response.json()
                return data.get("organic_results", [])[:3]
    except Exception as e:
        st.error(f"搜索失敗: {str(e)}")
        return []

async def async_api_call(session: aiohttp.ClientSession, prompt: str, config: Dict) -> str:
    """執行API調用"""
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config['temperature'] * 0.7 if config['deepthink'] else config['temperature'],
        "max_tokens": 500
    }
    try:
        async with session.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content']
            return f"API Error: {await response.text()}"
    except Exception as e:
        return f"Network Error: {str(e)}"

def generate_identity_pool(count: int, roles: List[str], styles: List[str], langs: List[str]) -> List[Dict]:
    """生成身份特征池"""
    from random import choices
    return [{
        "role": choices(roles)[0],
        "style": choices(styles)[0],
        "language": choices(langs)[0],
        "search_weight": 0.3 if choices([True, False], weights=[0.3, 0.7])[0] else 0,
        "prompt": f"用{choices(langs)[0]}口語，以{choices(styles)[0]}風格，從{choices(roles)[0]}的視角發表看法"
    } for _ in range(count)]

async def batch_generator(base_prompt: str, identities: List[Dict], config: Dict) -> pd.DataFrame:
    """批量生成引擎"""
    search_context = ""
    if config['websearch']:
        search_results = await web_search(base_prompt)
        search_context = "\n".join(
            [f"{idx+1}. {res['title']}: {res['snippet']}" 
            for idx, res in enumerate(search_results)]
        )

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for identity in identities:
            final_prompt = base_prompt
            if config['websearch'] and identity['search_weight'] > 0:
                final_prompt += f"\n[網絡背景]\n{search_context}\n"
            final_prompt += f"\n{identity['prompt']}"
            tasks.append(async_api_call(session, final_prompt, config))
        
        results = await asyncio.gather(*tasks)
    
    return pd.DataFrame([{
        "角色": identity["role"],
        "風格": identity["style"],
        "語言": identity["language"],
        "含搜索": "✅" if identity['search_weight'] > 0 else "❌",
        "回復": response,
        "生成時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } for identity, response in zip(identities, results)])

# ========================
# Streamlit UI
# ========================
def ui():
    st.set_page_config(page_title="AI水軍工廠Pro", layout="wide")
    st.title("🎭 AI水軍智能生成系統")
    
    # 初始化session狀態
    if 'custom_roles' not in st.session_state:
        st.session_state.custom_roles = []
    if 'custom_styles' not in st.session_state:
        st.session_state.custom_styles = []
    if 'custom_langs' not in st.session_state:
        st.session_state.custom_langs = []

    with st.sidebar:
        st.header("⚙️ 控制面板")
        
        # 生成設置
        count = st.slider("生成數量", 10, 500, 100, step=10)
        temperature = st.slider("創意程度", 0.0, 2.0, 0.7, step=0.1,
                              help="0: 保守回答, 2: 天馬行空")
        deepthink_enabled = st.checkbox("啟用DeepThink模式", 
                                      help="增強邏輯推理能力")
        websearch_enabled = st.checkbox("啟用網絡搜索",
                                      help="整合實時網絡信息（需要SerpAPI密鑰）")

        # 自定義類別管理
        with st.expander("🗃️ 類別管理"):
            # 新增角色
            new_role = st.text_input("新增角色", key="new_role")
            if st.button("➕ 添加角色"):
                if new_role and new_role not in DEFAULT_ROLES + st.session_state.custom_roles:
                    st.session_state.custom_roles.append(new_role)
            
            # 新增風格
            new_style = st.text_input("新增風格", key="new_style")
            if st.button("➕ 添加風格"):
                if new_style and new_style not in DEFAULT_STYLES + st.session_state.custom_styles:
                    st.session_state.custom_styles.append(new_style)
            
            # 新增語言
            new_lang = st.text_input("新增語言", key="new_lang")
            if st.button("➕ 添加語言"):
                if new_lang and new_lang not in DEFAULT_LANGS + st.session_state.custom_langs:
                    st.session_state.custom_langs.append(new_lang)

    # 主界面
    main_col = st.columns([4, 1])[0]
    with main_col:
        # 核心指令輸入
        base_prompt = st.text_area(
            "📝 核心指令輸入區",
            height=150,
            placeholder="例：請模擬香港Facebook論壇網友對新游戲《天月麻雀》的討論...",
            help="建議包括：\n• 討論主題\n• 需要強調的觀點\n• 特殊格式要求"
        )
        
        # 特征選擇
        st.subheader("🎨 特征選擇")
        col1, col2, col3 = st.columns(3)
        with col1:
            roles = st.multiselect(
                "選擇角色",
                DEFAULT_ROLES + st.session_state.custom_roles,
                default=["大學生", "上班族"]
            )
        with col2:
            styles = st.multiselect(
                "選擇風格",
                DEFAULT_STYLES + st.session_state.custom_styles,
                default=["幽默吐槽", "專業分析"]
            )
        with col3:
            langs = st.multiselect(
                "選擇語音",
                DEFAULT_LANGS + st.session_state.custom_langs,
                default=["粵語", "中英夾雜"]
            )

        # 生成控制
        if st.button("🚀 開始智能生成", type="primary", use_container_width=True):
            if not base_prompt:
                st.warning("請輸入核心指令！")
            elif not roles or not styles or not langs:
                st.warning("Pick one！")
            else:
                with st.status("生成進度", expanded=True) as status:
                    try:
                        # 生成身份池
                        identities = generate_identity_pool(
                            count, roles, styles, langs
                        )
                        
                        # 執行生成
                        progress = st.progress(0, text="初始化生成引擎...")
                        config = {
                            "temperature": temperature,
                            "deepthink": deepthink_enabled,
                            "websearch": websearch_enabled
                        }
                        
                        df = asyncio.run(batch_generator(base_prompt, identities, config))
                        progress.progress(100, "生成完成！")
                        
                        # 顯示結果
                        status.update(label="生成完成 ✅", state="complete")
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "生成時間": st.column_config.DatetimeColumn(
                                    format="YYYY-MM-DD HH:mm:ss"
                                )
                            }
                        )
                        
                        # 導出CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 下載CSV",
                            data=csv,
                            file_name=f"ai_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"生成失敗: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    ui()