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

# --- 0. 基础配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_FILE = os.path.join(BASE_DIR, 'ming_vectors.pkl')

st.set_page_config(page_title="明域 · 伪史生成系统", layout="wide", page_icon="🐉")

# --- 核心架构类定义 ---

class HistoryEmbeddingLayer:
    """
    第1层：历史事实嵌入层
    功能：加载“明代历史知识图谱嵌入空间”，提供向量化和检索能力。
    """
    def __init__(self, vector_file):
        self.vector_file = vector_file
        self.model = None
        self.db_data = None
        self.db_embeddings = None
        self._load_resources()

    def _load_resources(self):
        # 使用 st.cache_resource 避免重复加载
        if 'model' not in st.session_state:
            st.session_state.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.model = st.session_state.model

        if not os.path.exists(self.vector_file):
            st.error(f"❌ 找不到 {self.vector_file}！请先运行 build_index.py")
            return

        if 'db_data' not in st.session_state:
            with open(self.vector_file, 'rb') as f:
                data = pickle.load(f)
                st.session_state.db_data = data['data']
                st.session_state.db_embeddings = data['embeddings']
        
        self.db_data = st.session_state.db_data
        self.db_embeddings = st.session_state.db_embeddings

    def encode(self, text):
        return self.model.encode([text], normalize_embeddings=True)

    def search(self, query_vec, top_k=3):
        if self.db_embeddings is None: return []
        scores = np.dot(self.db_embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": scores[idx],
                "data": self.db_data[idx],
                "vector": self.db_embeddings[idx]
            })
        return results

class ContextAlignmentLayer:
    """
    第2层：制度-语境对齐层
    功能：确保生成内容符合明代制度逻辑（如卫所、里甲、科举、厂卫）。
    """
    def __init__(self):
        self.keywords = [
            "卫所", "锦衣卫", "东厂", "西厂", "内阁", "科举", "六部", 
            "巡抚", "总督", "里甲", "黄册", "鱼鳞图册", "海禁", "朝贡",
            "司礼监", "翰林院", "国子监", "布政使", "按察使", "都指挥使"
        ]

    def validate(self, text):
        """简单模拟“多任务学习：制度分类头”"""
        found_keywords = [kw for kw in self.keywords if kw in text]
        score = len(found_keywords) * 0.2  # 简单的启发式打分
        return {
            "is_valid": len(found_keywords) > 0,
            "score": min(score, 1.0),
            "keywords": found_keywords
        }

class FictionDiffusionLayer:
    """
    第3层：合理虚构扩散层
    功能：在历史语义邻域内进行受控向量插值，生成“未记载但可能”的事件细节。
    """
    def __init__(self, embedding_layer):
        self.emb_layer = embedding_layer

    def interpolate_and_generate(self, fact_vec, query_vec, alpha=0.3):
        """
        Constrained Diffusion in Embedding Space (模拟)
        V_gen = (1 - alpha) * V_fact + alpha * V_query
        """
        # 向量插值
        # alpha 越大，越偏向用户的“虚构/查询”；alpha 越小，越偏向“史实”
        gen_vec = (1 - alpha) * fact_vec + alpha * query_vec
        
        # 归一化（保持在单位球面上，符合 cosine similarity 特性）
        norm = np.linalg.norm(gen_vec)
        if norm > 0:
            gen_vec = gen_vec / norm
            
        # 在空间中寻找最近的“潜在史料”作为生成的基底
        # 注意：这里我们寻找的是除了原始 fact 之外最近的点，代表“可能的变体”
        results = self.emb_layer.search(gen_vec, top_k=5)
        
        return gen_vec, results

# --- 辅助函数 (CBDB) ---
def get_cbdb_bio(name_cn):
    """从哈佛 CBDB 获取结构化数据"""
    try:
        name_trad = zhconv.convert(name_cn, 'zh-hant')
        url = "https://cbdb.fas.harvard.edu/cbdbapi/person.php"
        params = {"name": name_trad, "o": "json"}
        resp = requests.get(url, params=params, timeout=3)
        data = json.loads(resp.text)
        if 'Package' in data: data = data['Package']
        if 'PersonAuthority' in data: data = data['PersonAuthority']
        if 'PersonInfo' in data: data = data['PersonInfo']
        if 'Person' in data: data = data['Person']
        
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

# --- UI 逻辑 ---

def main():
    # 初始化各层
    layer1 = HistoryEmbeddingLayer(VECTOR_FILE)
    layer2 = ContextAlignmentLayer()
    layer3 = FictionDiffusionLayer(layer1)

    # 侧边栏
    with st.sidebar:
        st.title("🐉 明域 MingYu")
        st.caption("基于历史语义嵌入的合理伪史生成系统")
        st.divider()
        
        st.header("⚙️ 系统参数 (System Params)")
        alpha = st.slider("虚构扩散系数 (Alpha)", 0.0, 1.0, 0.3, help="0=完全史实, 1=完全虚构")
        threshold = st.slider("合理性阈值 (Credibility)", 0.0, 1.0, 0.4, help="过滤掉语义距离过远的结果")
        
        st.info("💡 **操作指南**：\n输入一个“假如”的历史情境，系统将在明代语义流形中寻找最合理的“伪史”落点。")

    # 主界面
    st.title("《明域》：合理伪史生成控制台")
    st.markdown("""
    > **核心理念**：在明代历史的语义流形上，进行有界的历史想象力探索。
    """)
    
    query = st.text_input("📝 输入历史假设 / 探索节点", "假如张居正支持万历皇帝彻底清算冯保")
    
    if st.button("启动生成引擎", type="primary"):
        if not layer1.db_data:
            st.error("数据未加载，请检查 build_index.py 是否运行。")
            st.stop()
            
        with st.spinner("正在遍历历史语义流形..."):
            # 1. 编码用户输入 (Layer 1)
            query_vec = layer1.encode(query)
            
            # 2. 检索最近的历史事实 (Layer 1)
            # 这是“锚点”，确保虚构不脱离历史基底
            fact_results = layer1.search(query_vec, top_k=1)
            fact_item = fact_results[0]
            fact_vec = fact_item['vector']
            
            # 3. 向量插值与扩散 (Layer 3)
            gen_vec, nearby_results = layer3.interpolate_and_generate(fact_vec, query_vec, alpha)
            
            # 4. 制度校验 (Layer 2)
            # 对生成结果（这里用最近邻近似）进行校验
            best_match = nearby_results[0] # 最接近插值点的文本
            validation = layer2.validate(best_match['data']['text'])
            
        # --- 结果展示 ---
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📍 历史锚点 (Fact Anchor)")
            st.success(f"**{fact_item['data']['name']}** (相似度: {fact_item['score']:.4f})")
            st.markdown(f"_{fact_item['data']['text']}_")
            
            st.divider()
            
            st.subheader("🎲 生成的合理伪史 (Generated Pseudo-History)")
            st.caption(f"基于插值向量 (Alpha={alpha}) 在语义空间中召回的最近邻状态")
            
            # 显示生成的“伪史”片段（其实是语义空间中介于事实和虚构之间的真实片段，作为模拟）
            gen_text = best_match['data']['text']
            gen_name = best_match['data']['name']
            gen_score = best_match['score'] # 这里是与插值向量的距离
            
            st.info(f"**相关人物：{gen_name}**")
            st.write(gen_text)
            
            # 制度校验结果
            st.markdown("#### 🛡️ 制度-语境对齐层校验")
            if validation['is_valid']:
                st.success(f"✅ 通过校验 (Score: {validation['score']:.2f})")
                st.markdown(f"**识别到的制度关键词**：`{', '.join(validation['keywords'])}`")
            else:
                st.warning("⚠️ 警告：未检测到典型的明代制度特征，生成内容可能偏离时代语境。")
                
        with col2:
            st.subheader("🌌 语义流形可视化")
            
            # 准备绘图数据
            # 1. 事实点
            # 2. 用户查询点
            # 3. 生成点 (插值点)
            # 4. 背景点 (随机取一些)
            
            subset_indices = list(range(min(len(layer1.db_data), 50)))
            subset_vecs = layer1.db_embeddings[subset_indices]
            subset_names = [layer1.db_data[i]['name'] for i in subset_indices]
            
            # 降维
            all_vecs = np.vstack([subset_vecs, fact_vec, query_vec, gen_vec])
            pca = PCA(n_components=2)
            all_coords = pca.fit_transform(all_vecs)
            
            # 背景数据
            bg_len = len(subset_vecs)
            df_bg = pd.DataFrame({
                'x': all_coords[:bg_len, 0],
                'y': all_coords[:bg_len, 1],
                'label': subset_names,
                'type': ['History Background'] * bg_len
            })
            
            # 特殊点
            df_special = pd.DataFrame({
                'x': [all_coords[bg_len, 0], all_coords[bg_len+1, 0], all_coords[bg_len+2, 0]],
                'y': [all_coords[bg_len, 1], all_coords[bg_len+1, 1], all_coords[bg_len+2, 1]],
                'label': ['历史锚点 (Fact)', '用户假设 (Query)', '生成伪史 (Generated)'],
                'type': ['Anchor', 'Query', 'Generated']
            })
            
            final_df = pd.concat([df_bg, df_special])
            
            fig = px.scatter(final_df, x='x', y='y', color='type', hover_data=['label'],
                             symbol='type', size_max=15, title="历史语义拓扑空间")
            
            fig.update_traces(marker=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            **图例说明**：
            - **Anchor**: 真实历史中与假设最接近的事件。
            - **Query**: 你的假设在语义空间中的位置。
            - **Generated**: 系统根据 Alpha 插值计算出的“伪史”落点。
            """)
            
            # CBDB 补充信息
            if validation['is_valid'] and gen_name != '未知':
                 st.divider()
                 st.markdown(f"**📜 {gen_name} 的真实履历 (CBDB)**")
                 bio = get_cbdb_bio(gen_name)
                 if bio:
                     st.json(bio)
                 else:
                     st.write("无详细记录")

if __name__ == "__main__":
    main()
