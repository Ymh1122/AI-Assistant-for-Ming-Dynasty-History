# 1. 配置路径
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'#！！！！关梯子运行更快！！！
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --- 核心修改开始 ---
# 1. 获取当前脚本(build_index.py)所在的绝对路径
current_script_path = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出数据文件夹的绝对路径
# 这样无论你在终端哪个目录下运行，Python 都能精准找到桌面上这个文件夹
DATA_FOLDER = os.path.join(current_script_path, 'ming_dynasty_cn')

print(f"📍 锁定数据路径: {DATA_FOLDER}")
# --- 核心修改结束 ---


def read_and_chunk_files(folder_path, chunk_size=150):
    """
    读取文件夹下的所有txt，并按长度切分成小段
    chunk_size: 每段大约多少字
    """
    all_chunks = []
    
    # 查找所有 .txt 文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print(f"❌ 错误：在 '{folder_path}' 下没找到 .txt 文件！请检查文件夹名字。")
        return []

    print(f"📂 发现 {len(txt_files)} 个人物传记文件，开始处理...")

    for file_path in txt_files:
        # 从文件名提取人名 (例如 "ming_dynasty_bios/张居正.txt" -> "张居正")
        file_name = os.path.basename(file_path)
        person_name = file_name.replace('.txt', '')
        
        try:
            # 尝试 UTF-8 读取，如果报错尝试 GBK (防止 Windows 编码问题)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()

        # --- 切片逻辑 (Chunking) ---
        # 简单粗暴但有效：按句号拆分，然后拼凑成 chunk_size 大小的块
        sentences = content.replace('\n', '').split('。')
        
        current_chunk = ""
        for sent in sentences:
            if not sent.strip(): continue
            
            current_chunk += sent + "。"
            
            # 如果当前块够长了，就存起来，并开启新的一块
            if len(current_chunk) >= chunk_size:
                all_chunks.append({
                    "id": f"{person_name}_{len(all_chunks)}",
                    "name": person_name,
                    "text": current_chunk
                })
                current_chunk = "" # 重置
        
        # 处理最后剩余的一点点文本
        if current_chunk:
            all_chunks.append({
                "id": f"{person_name}_last",
                "name": person_name,
                "text": current_chunk
            })
            
    return all_chunks

def create_embeddings():
    # 1. 读取并切分数据
    wiki_data = read_and_chunk_files(DATA_FOLDER)
    
    if not wiki_data:
        return

    print(f"✅ 数据预处理完成！共切分为 {len(wiki_data)} 个文本片段。")
    print("⏳ 正在加载 BGE 模型 (第一次运行需要下载)...")
    
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    print("🚀 正在生成向量 (这可能需要几十秒)...")
    texts = [item["text"] for item in wiki_data]
    
    # normalize_embeddings=True 对计算余弦相似度非常重要
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    print(f"📊 向量生成完毕。维度: {embeddings.shape}")

    # 保存到本地
    output_file = 'ming_vectors.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'data': wiki_data, 'embeddings': embeddings}, f)
    
    print(f"💾 数据库已保存为: {output_file}")

if __name__ == "__main__":
    create_embeddings()