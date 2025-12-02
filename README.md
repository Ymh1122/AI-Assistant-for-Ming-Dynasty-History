
# 《明域》 (MingYu)

### 基于向量偏移的合理伪史生成系统

### Vector-Guided Reasonable Pseudo-History Generator

**《明域》** 是一个基于 **RAG (检索增强生成)** 与 **向量空间插值** 技术的实验性历史推演工具。

不同于传统大模型“天马行空”的虚构，本系统利用 Embedding 技术在“严谨史实”与“用户假设”之间构建一条**向量通道**。通过控制偏移系数 $\alpha$，我们在明代历史的语义空间中寻找“最合理的偏差邻域”，从而约束大模型生成既符合明代语境、又顺应用户假设的“伪史”。

**MingYu** is an experimental historical deduction tool based on **RAG** and **Vector Space Interpolation**.

Unlike the unconstrained hallucinations of traditional LLMs, MingYu constructs a **vector path** between "Strict History" and "User Hypothesis" using Embedding technology. By controlling the offset coefficient $\alpha$, we locate the "most plausible deviation neighborhood" within the semantic space of Ming Dynasty history, constraining the LLM to generate "pseudo-history" that fits the historical context while satisfying the user's "What-If" scenario.

-----

## 🧠 核心逻辑：Embedding 如何控制生成？

## Core Logic: How Embeddings Guide Generation

本项目的核心并非简单的关键词搜索，而是 **向量空间内的导航 (Vector Navigation)**。系统通过 `FictionDiffusionLayer` 实现以下逻辑：

The core is not simple keyword search, but **Vector Navigation**. The system implements the following logic via `FictionDiffusionLayer`:

$$V_{target} = (1 - \alpha) \cdot V_{fact} + \alpha \cdot V_{query}$$

1.  **定位锚点 (Anchor)**: 首先找到与用户假设 ($V_{query}$) 最接近的真实历史事件 ($V_{fact}$)。

2.  **向量插值 (Interpolation)**: 根据系数 $\alpha$ 计算目标向量 $V_{target}$。

      * $\alpha \to 0$: 结果趋向真实历史（复读史书）。
      * $\alpha \to 1$: 结果趋向用户假设（可能脱离时代背景）。

3.  **邻域检索 (Neighbor Retrieval)**: **关键步骤**。系统不直接使用用户的文本去搜索，而是使用计算出的 $V_{target}$ 在数据库中检索“在该平行时空下可能发生的相关事件”。

4.  **受控生成 (Constrained Generation)**: 将这些“偏移后的历史上下文”喂给 Qwen 大模型，使其在限定的语境下进行写作。

5.  **Anchor Positioning**: Find the real historical event ($V_{fact}$) closest to the user's hypothesis ($V_{query}$).

6.  **Vector Interpolation**: Calculate the target vector $V_{target}$ based on $\alpha$.

7.  **Neighbor Retrieval**: **Key Step**. Instead of searching with user text, the system uses $V_{target}$ to retrieve "relevant events that might happen in this parallel timeline."

8.  **Constrained Generation**: Feed these "shifted historical contexts" to the Qwen LLM for grounded writing.

-----

## ✨ 功能特性 / Features

  * **🛡️ 历史语义嵌入 (Historical Embeddings)**
      * 基于 `BAAI/bge-small-zh-v1.5` 模型，对《明史》及明代维基条目进行细粒度向量化。
      * Supports fine-grained vectorization of Ming Dynasty historical texts.
  * **🎛️ 动态伪史调节 (Dynamic Adjustment)**
      * 用户可通过滑块实时调节 $\alpha$ 值，直观感受“史实”与“虚构”的拉锯。
      * Adjust $\alpha$ in real-time to balance between historical accuracy and imagination.
  * **📊 语义空间可视化 (PCA Visualization)**
      * 使用 Plotly 展示历史背景点、锚点、查询点及生成点的空间分布关系。
      * Visualizes the spatial distribution of history, anchor, query, and generated points via PCA.
  * **⚖️ 制度一致性校验 (Institutional Consistency)**
      * 内置关键词校验器，检测生成内容是否包含“锦衣卫”、“内阁”、“六部”等明代特有制度名词。
      * Built-in validator ensures generated content contains Ming-specific institutional terms.

-----

## 🚀 快速开始 / Quick Start

### 1\. 环境准备 / Prerequisites

```bash
# 推荐使用 Python 3.8+
pip install -r requirements.txt
```

### 2\. 配置 API Key / Setup API Key

本项目使用 **Qwen-Plus (通义千问)** 进行文本生成。
请在项目根目录创建 `.env` 文件或设置环境变量：
This project uses **Qwen-Plus** for text generation. Create a `.env` file or set env variable:

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 3\. 构建数据索引 / Build Index

首次运行前，需处理 `ming_dynasty_cn/` 下的原始语料并生成向量数据库。
Process raw corpus and generate the vector database before the first run.

```bash
python build_index.py
# 输出: 💾 数据库已保存为: ming_vectors.pkl
```

### 4\. 启动系统 / Launch App

```bash
streamlit run app.py
```

-----

## 📂 项目结构 / Structure

```text
.
├── app.py                  # Streamlit 前端交互与可视化入口 (UI & Visualization)
├── core_logic.py           # 核心业务逻辑 (Vector Search, Interpolation, LLM Call)
├── build_index.py          # 离线数据处理与向量化脚本 (Data Processing & Embedding)
├── Data_preprocessing.py   # 维基百科爬虫 (Wikipedia Scraper)
├── ming_dynasty_cn/        # 原始语料库 (Raw Corpus)
└── ming_vectors.pkl        # 预计算的向量数据库 (Pre-computed Vector DB)
```
-----
## ⚠️ 免责声明 / Disclaimer

本项目生成内容均为基于算法的虚构文本（伪史），仅供数字人文研究与娱乐，请勿引用为真实历史资料。
Generated content is algorithmically fictional (pseudo-history). Do not cite as real historical facts.
