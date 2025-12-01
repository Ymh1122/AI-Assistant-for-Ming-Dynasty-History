# 《明域》：基于历史语义嵌入的合理伪史生成系统
# MingYu: Reasonable Pseudo-History Generation System Based on Historical Semantic Embeddings

**《明域》** 是一个融合数字人文与自然语言处理技术的创新实验平台，旨在通过 **历史文本嵌入 (Historical Text Embedding)** 构建明代历史的语义拓扑空间，并在此空间中生成具有高度可信度的“合理伪史”。

**MingYu** is an innovative experimental platform integrating Digital Humanities and Natural Language Processing (NLP). It aims to construct a semantic topological space of Ming Dynasty history using **Historical Text Embedding** and generate highly credible "reasonable pseudo-history" within this space.

---

## 💡 核心理念 / Core Philosophy

> **在明代历史的语义流形上，进行有界的历史想象力探索。**
> **Exploring bounded historical imagination on the semantic manifold of Ming Dynasty history.**

与传统大模型自由创作不同，《明域》将 Embedding 技术从 **表示工具** 升级为 **生成约束机制**。虚构内容的生成并非凭空想象，而是在历史语义流形的局部邻域内进行 **受约束的向量探索**。

Unlike traditional LLM free-form creation, MingYu upgrades Embedding technology from a **representation tool** to a **generation constraint mechanism**. The generation of fictional content is not baseless imagination but **constrained vector exploration** within the local neighborhood of the historical semantic manifold.

---

## 🏗️ 技术架构 / Technical Architecture

本系统采用 **三层嵌入体系 (Three-Layer Embedding System)**：

### 1. 历史事实嵌入层 (Historical Fact Embedding Layer)
- **功能**：对《明实录》《明史》等正史文本进行细粒度向量化，构建“明代历史知识图谱嵌入空间”。
- **实现**：使用 Sentence-BERT 模型 (BAAI/bge-small-zh) 进行语义编码。
- **Function**: Performs fine-grained vectorization of official historical texts (e.g., "Ming Shilu", "History of Ming") to construct the "Ming Dynasty Historical Knowledge Graph Embedding Space".
- **Implementation**: Uses Sentence-BERT (BAAI/bge-small-zh) for semantic encoding.

### 2. 制度-语境对齐层 (Institution-Context Alignment Layer)
- **功能**：确保生成内容符合明代制度逻辑（如卫所、里甲、科举、厂卫）与时代语境。
- **实现**：基于关键词库的制度逻辑校验与评分机制。
- **Function**: Ensures generated content aligns with Ming Dynasty institutional logic (e.g., Wei-Suo system, Lijia system, Imperial Examinations, Eastern/Western Depot) and historical context.
- **Implementation**: Institutional logic validation and scoring mechanism based on keyword dictionaries.

### 3. 合理虚构扩散层 (Reasonable Fiction Diffusion Layer)
- **功能**：在历史语义邻域内进行受控向量插值，生成“未记载但可能”的事件细节。
- **实现**：向量空间插值 (Vector Interpolation) + 最近邻检索 (Nearest Neighbor Search)。
- **Function**: Performs controlled vector interpolation within the historical semantic neighborhood to generate "unrecorded but plausible" event details.
- **Implementation**: Vector Interpolation + Nearest Neighbor Search.

---

## 🚀 快速开始 / Quick Start

### 1. 环境准备 / Prerequisites
确保已安装 Python 3.8+。
Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

### 2. 构建历史语义索引 / Build Historical Semantic Index
首次运行前，需要处理原始文本并生成向量数据库。
Before the first run, process raw texts and generate the vector database.

```bash
python build_index.py
```
> 成功后会生成 `ming_vectors.pkl` 文件。
> This will generate the `ming_vectors.pkl` file upon success.

### 3. 启动系统 / Launch System
启动 Streamlit Web 界面。
Launch the Streamlit Web Interface.

```bash
streamlit run app.py
```

---

## 📂 文件结构 / File Structure

```text
.
├── app.py                  # 主应用程序 (Main Application - Streamlit)
├── build_index.py          # 索引构建脚本 (Index Building Script)
├── ming_dynasty_cn/        # 原始历史语料 (Raw Historical Corpus - .txt)
├── ming_vectors.pkl        # 向量数据库 (Vector Database - Generated)
├── requirements.txt        # 依赖列表 (Dependencies)
└── README.md               # 说明文档 (Documentation)
```

---

## 🖼️ 系统预览 / System Preview

- **历史锚点 (Fact Anchor)**: 真实历史中与假设最接近的事件。
- **语义流形可视化**: 通过 PCA 降维展示历史事件与虚构假设在语义空间中的分布。
- **制度校验**: 自动检测生成内容是否符合明代政治制度特征。

---

## 👥 致谢 / Credits

本项目受马伯庸“在历史缝隙中讲故事”的启发，旨在为历史教学、公众史学与文化创作提供一种新型的认知工具。

Inspired by Ma Boyong's concept of "telling stories in the cracks of history," this project aims to provide a new cognitive tool for history education, public history, and cultural creation.
