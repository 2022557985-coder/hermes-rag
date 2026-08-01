# Hermes-RAG

轻量级、高精度 RAG 检索优化框架，专为本地化部署设计。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概述

Hermes-RAG 是一套可完全本地化部署的 RAG（检索增强生成）框架，在 **8GB RAM + 4核CPU** 的笔记本环境下，将检索命中率从默认的 65% 以下提升至 **90% 以上**。

### 核心特性

- **层级化分块**：标题层级切分 + 语义分割，保留文档结构信息
- **双索引架构**：稠密向量（ChromaDB/HNSW）+ 稀疏 BM25 并行检索
- **RRF 动态权重融合**：根据查询类型（产品型号/口语化）自动调整稠密/稀疏权重
- **Cross-Encoder 重排序**：BGE-Reranker 精排，提升 Top-K 精度
- **查询扩展**：同义词扩展 + 可选 HyDE（假设文档嵌入）
- **规则引擎**：章节预过滤，支持产品型号等模式匹配
- **语义缓存**：相似查询命中缓存，降低延迟
- **多格式支持**：PDF、DOCX、PPTX、TXT、Markdown、网页
- **资源友好**：CPU 推理、内存优化、SQLite 降级存储
- **安全防护**：路径遍历保护、SSRF 防护、文件大小限制、API Key 认证
- **生产就绪**：Docker 支持、速率限制、优雅关闭、线程安全配置

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    用户查询                           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  ① 查询扩展 (QueryExpander)                          │
│     同义词扩展 + HyDE (可选)                          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  ② 并行多路召回 (Parallel Retrieval)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Dense (HNSW) │  │ BM25 (稀疏)  │  │ Rule 引擎  │  │
│  │   Top-100    │  │   Top-100    │  │ 章节过滤   │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼─────────────────┼───────────────┼────────┘
          └─────────┬───────┘               │
                    ▼                       │
┌─────────────────────────────────────────────────────┐
│  ③ RRF 动态权重融合 (RRFFusion)                       │
│     稠密/稀疏加权 → Top-50                            │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  ④ Cross-Encoder 重排序 (BGE-Reranker)               │
│     候选精排 → Top-5                                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  ⑤ 结果返回 + 可选 LLM 生成                           │
└─────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- 8GB+ RAM（推荐 16GB）
- Windows / macOS / Linux

### 安装

```bash
# 克隆项目
cd hermes_rag

# 安装依赖（推荐使用 Poetry）
pip install poetry
poetry install

# 或使用 pip
pip install -e .
```

### Docker 部署

```bash
# 构建镜像
docker build -t hermes-rag .

# 运行容器
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e HERMES_API_KEY=your-secret-key \
  hermes-rag
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件（可选，默认配置已可用）
# config.yaml
```

### 首次运行

首次运行时会自动下载模型（约 130MB），请保持网络畅通。

```bash
# 摄入文档
python cli.py ingest ./evaluation/data/sample_docs/

# 查询
python cli.py query "什么是机器学习？"

# 启动 API 服务
python cli.py serve

# 启动 Web 界面
python cli.py ui
```

## CLI 使用

```bash
# 摄入文档
python cli.py ingest <路径或URL>        # 摄入单个文件或整个目录

# 查询
python cli.py query "你的问题"           # 默认返回 Top-5
python cli.py query "你的问题" --top-k 10  # 返回 Top-10
python cli.py query "你的问题" --no-reranker  # 禁用重排序
python cli.py query "你的问题" --format json  # JSON 格式输出

# 启动服务
python cli.py serve                      # 启动 FastAPI 服务 (端口 8000)
python cli.py ui                         # 启动 Gradio Web 界面 (端口 7860)
```

## API 接口

启动服务后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 认证

可通过环境变量 `HERMES_API_KEY` 启用 API Key 认证。启用后需要在请求头中携带 `X-API-Key`。

### 健康检查

```
GET /api/v1/health
```

### 文档摄入

```
POST /api/v1/ingest
{
  "source": "/path/to/document.pdf"
}
```

支持的文件类型：`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.csv`, `.json`

### 检索查询

```
POST /api/v1/query
{
  "query": "什么是机器学习？",
  "top_k": 5,
  "use_reranker": true,
  "generate_answer": false
}
```

## 配置说明

所有参数通过 `config.yaml` 管理，无需修改代码：

| 模块 | 关键参数 | 说明 |
|------|---------|------|
| embedding | model_name, device | 嵌入模型选择与设备 |
| chunking | chunk_size: 512, overlap: 128 | 分块大小与重叠 |
| chromadb | hnsw_ef_construction, hnsw_M | HNSW 索引参数 |
| bm25 | b, k1 | BM25 算法参数 |
| retrieval | dense_top_k, sparse_top_k, rrf_k | 检索参数 |
| reranking | enabled, model_name, timeout | 重排序配置 |
| generation | provider, model | LLM 生成配置 |

## 评估

```bash
# 运行评估
python -m evaluation.eval

# 基线对比
python -m evaluation.eval --baseline

# 输出 JSON 结果
python -m evaluation.eval --output results.json
```

### 评估指标

- **Hit Rate@K**（K=1,3,5）：Top-K 结果中命中相关文档的比例
- **MRR**（Mean Reciprocal Rank）：第一个相关文档排名的倒数均值
- **NDCG@10**：归一化折损累计增益

## 项目结构

```
hermes_rag/
├── cli.py                          # 命令行入口
├── config.yaml                     # 全局配置
├── pyproject.toml                  # 依赖管理
│
├── src/
│   ├── config.py                   # Pydantic 配置加载
│   ├── core/
│   │   ├── ingestion/              # 文档解析（PDF/DOCX/PPTX/TXT/Web）
│   │   ├── chunking/               # 层级化分块 + 语义分割
│   │   ├── indexing/               # ChromaDB + BM25 双索引
│   │   ├── retrieval/              # 多路召回 + RRF 融合
│   │   ├── reranking/              # Cross-Encoder 重排序
│   │   └── generation/             # LLM 客户端（OpenAI/Ollama）
│   └── utils/                      # 日志、缓存、计时、内存监控
│
├── api/                            # FastAPI 服务
├── ui/                             # Gradio Web 界面
├── tests/                          # 测试套件
├── evaluation/                     # 评估脚本与数据集
│   └── data/
│       ├── ground_truth.json       # 基准真值
│       └── sample_docs/            # 示例文档
└── docs/                           # 技术文档
```

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 向量数据库 | ChromaDB | 无独立进程、原生 Python、低内存 |
| Embedding 模型 | bge-small-en-v1.5 | 384维、CPU推理 50ms/句 |
| 稀疏检索 | rank_bm25 | 纯 Python、无外部依赖 |
| 重排序 | bge-reranker-base | 200MB、精度提升显著 |
| 融合算法 | 改进 RRF | 动态权重、k=60 |
| LLM 接口 | OpenAI/Ollama | 可插拔、支持本地部署 |

## License

MIT