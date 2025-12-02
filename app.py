import os
# 必须在导入 sentence_transformers 之前设置环境变量，否则镜像源可能不生效
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 抑制 TensorFlow 日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import dashscope
from sklearn.decomposition import PCA

# Import core logic
from core_logic import (
    HistoryEmbeddingLayer,
    ContextAlignmentLayer,
    FictionDiffusionLayer,
    QwenGenerationLayer,
    ContentAuditor,
    ExternalKnowledgeLayer
)

# --- 0. 基础配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_FILE = os.path.join(BASE_DIR, 'ming_vectors.pkl')

# 加载 API Key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv('DASHSCOPE_API_KEY')
if api_key:
    dashscope.api_key = api_key
else:
    st.warning("⚠️ 未检测到 DASHSCOPE_API_KEY，请在 .env 文件中配置，否则无法使用 Qwen 生成文本。")

st.set_page_config(page_title="明域 · 伪史生成系统", layout="wide", page_icon="🐉")

# --- UI 逻辑 ---

def main():
    # 初始化各层
    layer1 = HistoryEmbeddingLayer(VECTOR_FILE)
    layer2 = ContextAlignmentLayer()
    layer3 = FictionDiffusionLayer(layer1)
    layer4 = QwenGenerationLayer()
    auditor = ContentAuditor()

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
            
        with st.spinner("正在遍历历史语义流形并生成伪史..."):
            # 1. 编码用户输入 (Layer 1)
            query_vec = layer1.encode(query)
            
            # 2. 检索最近的历史事实 (Layer 1)
            # 这是“锚点”，确保虚构不脱离历史基底
            fact_results = layer1.search(query_vec, top_k=1)
            fact_item = fact_results[0]
            fact_vec = fact_item['vector']
            
            # 3. 向量插值与扩散 (Layer 3)
            # 传入 exclude_id，确保不返回史实本身
            gen_vec, nearby_results = layer3.interpolate_and_generate(
                fact_vec, 
                query_vec, 
                alpha, 
                exclude_id=fact_item['data']['id']
            )
            
            # 4. 制度校验 (Layer 2)
            # 对生成结果（这里用最近邻近似）进行校验
            best_match = nearby_results[0] # 最接近插值点的文本
            validation = layer2.validate(best_match['data']['text'])
            
            # 5. 大模型生成 (Layer 4 - NEW)
            # 提取 context 文本列表
            nearby_texts = [r['data']['text'] for r in nearby_results]
            generated_pseudo_history = layer4.generate(
                query, 
                fact_item['data']['text'], 
                nearby_texts, 
                alpha
            )
            
            # 6. 双重审核 (Auditor)
            # 审核的是大模型生成的文本，而不是检索到的文本
            audit_result = auditor.audit(query, generated_pseudo_history)
            
        # --- 结果展示 ---
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(" 历史锚点 (Fact Anchor)")
            st.success(f"**{fact_item['data']['name']}** (相似度: {fact_item['score']:.4f})")
            st.markdown(f"_{fact_item['data']['text']}_")
            
            st.divider()
            
            st.subheader(" 生成的合理伪史 (Qwen Generated Pseudo-History)")
            st.caption(f"基于插值向量 (Alpha={alpha}) + Qwen-Plus 生成")
            
            # 显示生成的“伪史”
            st.markdown(generated_pseudo_history)
            
            # 制度校验结果
            st.markdown("####  Layer 2: 制度-语境对齐校验")
            # 对大模型生成的文本进行校验
            gen_validation = layer2.validate(generated_pseudo_history)
            
            if gen_validation['is_valid']:
                st.success(f" 通过校验 (Score: {gen_validation['score']:.2f})")
                st.markdown(f"**识别到的制度关键词**：`{', '.join(gen_validation['keywords'])}`")
            else:
                st.warning("⚠️ 警告：未检测到典型的明代制度特征，生成内容可能偏离时代语境。")
                
            # 双重审核结果
            st.markdown("####  Double Review: 内容合规性审核")
            if audit_result['passed']:
                st.success(f"✅ {audit_result['message']}")
            else:
                st.error(f"❌ {audit_result['message']}")
                st.caption("建议：调整 Alpha 值或细化指令以匹配已有史料库。")
                
        with col2:
            st.subheader(" 语义流形可视化")
            
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
            # 只有当条目被归类为“人物”时才调用 CBDB，避免用事件名去查人名数据库
            category = best_match['data'].get('category', '人物') # 兼容旧数据，默认为人物
            gen_name = best_match['data']['name']
            
            if validation['is_valid'] and gen_name != '未知' and category == '人物':
                 st.divider()
                 st.markdown(f"** {gen_name} 的真实履历 (CBDB)**")
                 bio = ExternalKnowledgeLayer.get_cbdb_bio(gen_name)
                 if bio:
                     st.json(bio)
                 else:
                     st.write("无详细记录")
            elif category != '人物':
                st.divider()
                st.info(f"ℹ 当前条目类别为 **{category}**，不展示人物履历。")

if __name__ == "__main__":
    main()
