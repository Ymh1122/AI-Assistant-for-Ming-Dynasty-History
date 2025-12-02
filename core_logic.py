import os
import pickle
import numpy as np
import requests
import json
import zhconv
import dashscope
import jieba
from http import HTTPStatus
from sentence_transformers import SentenceTransformer
import streamlit as st # Needed for st.cache_resource and st.session_state

class HistoryEmbeddingLayer:
    """
    Layer 1: Historical Fact Embedding Layer
    Function: Loads "Ming Dynasty Historical Knowledge Graph Embedding Space", providing vectorization and retrieval capabilities.
    """
    def __init__(self, vector_file):
        self.vector_file = vector_file
        self.model = None
        self.db_data = None
        self.db_embeddings = None
        self._load_resources()

    def _load_resources(self):
        # Use st.cache_resource to avoid reloading
        if 'model' not in st.session_state:
            st.session_state.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.model = st.session_state.model

        if not os.path.exists(self.vector_file):
            # In a real app, we might raise an error or log it, but for Streamlit we use st.error
            # However, to keep this logic pure, we should probably just print or raise exception
            # But since it relies on st.session_state, it's tied to Streamlit.
            # We will keep the st.error for now as it is tightly coupled.
            if 'st' in globals():
                st.error(f"Cannot find {self.vector_file}! Please run build_index.py first.")
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
    Layer 2: Institution-Context Alignment Layer
    Function: Ensures generated content aligns with Ming Dynasty institutional logic.
    """
    def __init__(self):
        self.keywords = [
            "卫所", "锦衣卫", "东厂", "西厂", "内阁", "科举", "六部", 
            "巡抚", "总督", "里甲", "黄册", "鱼鳞图册", "海禁", "朝贡",
            "司礼监", "翰林院", "国子监", "布政使", "按察使", "都指挥使"
        ]

    def validate(self, text):
        """Simple simulation of 'Multi-task Learning: Institution Classification Head'"""
        found_keywords = [kw for kw in self.keywords if kw in text]
        score = len(found_keywords) * 0.2  # Simple heuristic scoring
        return {
            "is_valid": len(found_keywords) > 0,
            "score": min(score, 1.0),
            "keywords": found_keywords
        }

class FictionDiffusionLayer:
    """
    Layer 3: Reasonable Fiction Diffusion Layer
    Function: Performs controlled vector interpolation within historical semantic neighborhood.
    """
    def __init__(self, embedding_layer):
        self.emb_layer = embedding_layer

    def interpolate_and_generate(self, fact_vec, query_vec, alpha=0.3, exclude_id=None):
        """
        Constrained Diffusion in Embedding Space (Simulation)
        V_gen = (1 - alpha) * V_fact + alpha * V_query
        """
        # Vector interpolation
        gen_vec = (1 - alpha) * fact_vec + alpha * query_vec
        
        # Normalization
        norm = np.linalg.norm(gen_vec)
        if norm > 0:
            gen_vec = gen_vec / norm
            
        # Search for nearest "potential historical records"
        results = self.emb_layer.search(gen_vec, top_k=10)
        
        # Exclude the anchor itself
        if exclude_id:
            filtered_results = [r for r in results if r['data']['id'] != exclude_id]
            return gen_vec, filtered_results
            
        return gen_vec, results

class QwenGenerationLayer:
    """
    Layer 4: LLM Generation Layer
    Function: Generates pseudo-history text using Qwen based on interpolated context.
    """
    def __init__(self):
        pass
        
    def generate(self, query, fact_text, nearby_texts, alpha):
        """
        Call Qwen to generate pseudo-history
        """
        if not dashscope.api_key:
            return "⚠️ API Key not configured, cannot generate text."
            
        context_str = "\n".join([f"- {t}" for t in nearby_texts[:3]])

        # Extract keywords to enforce their presence in generation
        words = list(jieba.cut(query))
        keywords = [w for w in words if len(w) > 1 and w not in ['假如', '如果', '支持', '反对', '彻底', '清算', '对于', '关于', '是否', '可以']]
        keywords_str = ", ".join(keywords)
        
        prompt = f"""
你是一个精通明代历史的小说家和历史学家。请根据以下信息，撰写一段“合理的伪史”。

【用户假设 (Query)】：{query}
【历史基底 (Fact)】：{fact_text}
【相关历史语境 (Context)】：
{context_str}

【生成要求】：
1. 这是一个“平行时空”的推演。虚构程度系数 Alpha = {alpha} (0=完全照搬史实，1=完全放飞想象)。
2. 当前 Alpha = {alpha}，请根据此系数平衡“史实”与“虚构”。
   - 如果 Alpha 较小，请尽量贴合原史，只做微调。
   - 如果 Alpha 较大，请在符合明代制度/人物性格的前提下，进行大胆的推演。
3. 必须模仿《明实录》或《明史》的笔法（文白相杂，庄重简练）。
4. 篇幅控制在 200 字左右。
5. 重点展示“用户假设”如何改变了历史走向，内容必须是虚构的，但要让人觉得“可能真实发生过”。
6. 【重要】为了保证内容与指令的一致性，请务必在文中包含以下关键词：{keywords_str}。

请直接开始撰写正文：
"""
        
        try:
            response = dashscope.Generation.call(
                dashscope.Generation.Models.qwen_plus,
                prompt=prompt,
                temperature=0.7,
                top_p=0.85
            )
            
            if response.status_code == HTTPStatus.OK:
                return response.output.text
            else:
                return f"Generation failed: {response.code} - {response.message}"
                
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

class ContentAuditor:
    """
    Double Review Mechanism
    Function: Validates if generated content matches user instructions and performs compliance checks.
    """
    def __init__(self):
        pass
        
    def audit(self, query, generated_text):
        """
        Audit if generated content matches Query intent
        """
        # 1. Entity Consistency Check
        # Use jieba for simple tokenization
        words = list(jieba.cut(query))
        query_keywords = [w for w in words if len(w) > 1 and w not in ['假如', '如果', '支持', '反对', '彻底', '清算', '对于', '关于']]
        
        missing_entities = []
        for kw in query_keywords:
            if kw not in generated_text:
                missing_entities.append(kw)
                
        if missing_entities:
            return {
                "status": "Warning",
                "message": f"Generated content may not fully cover entities in instruction: {', '.join(missing_entities)}",
                "passed": False
            }
            
        return {
            "status": "Pass",
            "message": "Content entity consistency check passed",
            "passed": True
        }

class ExternalKnowledgeLayer:
    """
    External Knowledge Layer
    Function: Fetches data from external sources like CBDB.
    """
    @staticmethod
    def get_cbdb_bio(name_cn):
        """Fetch structured data from Harvard CBDB"""
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
