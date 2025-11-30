import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import requests
import json
import zhconv
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# --- 0. 基础配置 (解决网络和路径问题) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 确保模型加载不卡顿
#pip install streamlit pandas plotly scikit-learn#！！关梯子运行！！！

# 获取当前脚本所在路径，确保能找到 .pkl 文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_FILE = os.path.join(BASE_DIR, 'ming_vectors.pkl')

st.set_page_config(page_title="明史 · 语义检索系统", layout="wide", page_icon="📜")

# --- 1. 核心资源加载 (带缓存，只跑一次) ---
@st.cache_resource
def load_resources():
    st.toast("正在加载 Embedding 模型和向量库...", icon="⏳")
    
    # 加载模型
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    # 加载数据
    if not os.path.exists(VECTOR_FILE):
        st.error(f"❌ 找不到 {VECTOR_FILE}！请先运行 build_index.py")
        return None, None, None
        
    with open(VECTOR_FILE, 'rb') as f:
        data = pickle.load(f)
        
    return model, data['data'], data['embeddings']

model, db_data, db_embeddings = load_resources()

# --- 2. CBDB API 函数 (我们之前调试完美的版本) ---
def get_cbdb_bio(name_cn):
    """从哈佛 CBDB 获取结构化数据"""
    name_trad = zhconv.convert(name_cn, 'zh-hant')
    url = "https://cbdb.fas.harvard.edu/cbdbapi/person.php"
    params = {"name": name_trad, "o": "json"}
    
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = json.loads(resp.text)
        
        # 剥洋葱逻辑
        if 'Package' in data: data = data['Package']
        if 'PersonAuthority' in data: data = data['PersonAuthority']
        if 'PersonInfo' in data: data = data['PersonInfo']
        if 'Person' in data: data = data['Person']
        
        # 归一化处理
        if isinstance(data, dict): target = data
        elif isinstance(data, list): target = data[0]
        else: return None
        
        basic = target.get('BasicInfo', {})
        return {
            "name": basic.get('ChName', name_cn),
            "birth": basic.get('YearBirth', '?'),
            "death": basic.get('YearDeath', '?'),
            "dynasty": basic.get('Dynasty', '明'),
            "native": basic.get('IndexAddr', '未知'),
            "id": basic.get('PersonId', 'N/A')
        }
    except:
        return None

# --- 3. 语义搜索逻辑 ---
def semantic_search(query, top_k=3):
    # 1. 问题转向量
    query_vec = model.encode([query], normalize_embeddings=True)
    # 2. 计算相似度
    scores = np.dot(db_embeddings, query_vec.T).flatten()
    # 3. 排序
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": scores[idx],
            "data": db_data[idx]
        })
    return results, query_vec

# --- 4. 界面 UI 布局 ---

# 标题栏
st.title("📜 明史 AI 语义检索系统")
st.markdown("结合 **NLP Embeddings** 与 **CBDB 数据库** 的数字人文探索项目")
st.divider()

# 侧边栏：搜索控制
with st.sidebar:
    st.header("🔍 探索面板")
    user_query = st.text_input("输入你的问题", "张居正和戚继光是什么关系？")
    
    st.info("💡 试一试：\n1. 谁是明朝开国皇帝？\n2. 嘉靖皇帝是否沉迷丹药？\n3. 徐霞客去过哪里？\n4. 土木堡之变")
    
    search_btn = st.button("开始分析", type="primary")
    
    st.divider()
    st.caption("Developed by CS Year 2 Group")

# 主界面逻辑
if search_btn or user_query:
    if not db_data:
        st.stop()
        
    # --- A. 执行搜索 ---
    results, query_vec = semantic_search(user_query)
    
    # 布局：左边显示文本结果，右边显示可视化
    col_left, col_right = st.columns([1.2, 1])
    
    # --- 左侧：检索结果 ---
    with col_left:
        st.subheader("📖 史料检索 (Retrieval)")
        
        # 提取排名第一的人名，用于查 CBDB
        top_person_name = results[0]['data']['name']
        
        for i, res in enumerate(results):
            score = res['score']
            text = res['data']['text']
            name = res['data']['name']
            
            # 动态卡片颜色
            border_color = "red" if i == 0 else "grey"
            
            with st.container(border=True):
                st.markdown(f"**Top {i+1} | {name}** (置信度: `{score:.4f}`)")
                st.markdown(f"> {text}")

    # --- 右侧：CBDB + 可视化 ---
    with col_right:
        # 1. CBDB 档案卡片
        st.subheader("🪪 人物档案 (CBDB API)")
        
        # 只有当置信度比较高时，才去查 CBDB，节省 API 资源
        if results[0]['score'] > 0.4:
            with st.spinner(f"正在连接哈佛服务器查询 {top_person_name}..."):
                bio = get_cbdb_bio(top_person_name)
            
            if bio:
                st.success(f"已找到 **{top_person_name}** 的官方记录")
                col_a, col_b = st.columns(2)
                col_a.metric("生卒年", f"{bio['birth']} - {bio['death']}")
                col_a.metric("籍贯", bio['native'])
                col_b.metric("CBDB ID", bio['id'])
                col_b.metric("朝代", bio['dynasty'])
            else:
                st.warning(f"CBDB 暂无 {top_person_name} 的结构化数据 (或网络超时)")
        else:
            st.info("未检测到明确的历史人物，暂不调用 CBDB。")

        # 2. 向量空间散点图 (亮点!)
        st.divider()
        st.subheader("🌌 语义空间可视化 (PCA)")
        
        # 准备绘图数据
        # 我们把数据库里的前 50 条拿出来画，太多会乱
        subset_indices = list(range(min(len(db_data), 50)))
        subset_vecs = db_embeddings[subset_indices]
        subset_names = [db_data[i]['name'] for i in subset_indices]
        subset_texts = [db_data[i]['text'][:30] for i in subset_indices]
        
        # 把用户的查询向量也加进去
        all_vecs = np.vstack([subset_vecs, query_vec])
        
        # PCA 降维到 2D
        pca = PCA(n_components=2)
        all_coords = pca.fit_transform(all_vecs)
        
        # 构建 DataFrame
        df = pd.DataFrame({
            'x': all_coords[:-1, 0],
            'y': all_coords[:-1, 1],
            'name': subset_names,
            'desc': subset_texts,
            'type': ['History'] * len(subset_names)
        })
        
        # 添加用户查询点
        query_df = pd.DataFrame({
            'x': [all_coords[-1, 0]],
            'y': [all_coords[-1, 1]],
            'name': ['YOUR QUERY'],
            'desc': [user_query],
            'type': ['Query']
        })
        
        final_df = pd.concat([df, query_df])
        
        # Plotly 画图
        fig = px.scatter(final_df, x='x', y='y', color='name', symbol='type',
                         hover_data=['desc'], size_max=15, 
                         title="语义距离分布图")
        # 标记出 Query 点为大星星
        fig.update_traces(marker=dict(size=12))
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✨ 图中距离越近的点，表示语义（含义）越相似。红星代表你的问题。")

else:
    st.write("👈 请在左侧侧边栏输入问题并点击“开始分析”")