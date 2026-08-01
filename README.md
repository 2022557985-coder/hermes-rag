# Hermes-RAG

轻量级、可本地化部署的企业级 RAG（检索增强生成）框架，面向 8GB RAM / 4 核 CPU 的开发与生产环境。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-0f766e.svg)](.github/workflows/ci.yml)

## 实测性能

以下为当前评测集（40 条查询、11 篇文档、62 个 chunk）在 `BAAI/bge-m3` 嵌入模型下的可复现结果，完整报告见 [docs/evaluation_report.md](docs/evaluation_report.md)。

| 检索变体 | H@1 | H@3 | H@5 | 文档H@5 | MRR | NDCG@10 | 平均延迟 |
|---------|------|------|------|---------|------|---------|---------|
| Dense baseline | 80.0% | 95.0% | 100.0% | 100.0% | 0.8821 | 0.8995 | 391ms |
| Dense + BM25 RRF（默认） | 80.0% | 97.5% | 100.0% | 100.0% | 0.8883 | 0.9054 | 415ms |
| RRF + 启发式重排 | 80.0% | 97.5% | 100.0% | 100.0% | 0.8883 | 0.9054 | 417ms |

难度拆解（默认 RRF 变体）: easy H@5 100.0% / medium H@5 100.0% / hard H@5 100.0%，40 条查询 Top-5 全部命中，chunk 级与文档级失败查询均为 0。

> 说明：评测集以概念检索为主，上述数字用于版本回归对比与质量门槛验证；生产环境建议扩充领域评测集后以 `run_eval.py` 输出为准。

## 核心特性

- 层级化分块：标题层级切分 + 语义分割，章节标题路径写入 chunk 文本，短小节不再被误丢
- 双索引架构：稠密向量（ChromaDB/HNSW）与稀疏 BM25 并行召回
- 混合查询扩展：稠密走原始查询，稀疏走同义词扩展查询，避免扩展噪声破坏向量排序
- RRF 动态权重融合：按查询类型（产品型号 / 口语化 / 默认）自动调整稠密/稀疏权重
- 文档存储（DocumentStore）：SQLite 持久化 chunk，向量维度或索引版本变化时自动重建
- BM25 SQLite 持久化：稀疏索引跨进程保留，`rank_bm25` 缺失时有纯 Python 回退
- 保守的重排序策略：默认关闭，启发式重排器已修复分数尺度不一致问题
- 查询扩展：ML 领域同义词扩展 + 可选 HyDE
- 语义缓存：相似查询命中缓存，降低延迟
- 多格式摄入：PDF、DOCX、PPTX、TXT、Markdown、网页
- 安全防护：路径遍历保护、CSRF 防护、文件大小限制、API Key 认证
- 生产就绪：FastAPI、Gradio 多标签页控制台、Docker、限流、优雅关闭

## 架构

```
用户查询 → 查询扩展 → 并行多路召回 → RRF 融合 → 可选重排 → 结果返回
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Dense (HNSW)        BM25 (SQLite)      DocumentStore
   ChromaDB            rank_bm25/回退     SQLite 事实源
   原始查询              扩展查询
```

三个存储由 `IndexManager` 协调：文档入库时并行写入向量库、BM25 与文档存储；启动时执行索引健康检查，向量维度变化、稀疏索引为空或索引版本升级时从文档存储自动重建。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 摄入文档（文件或目录）
python cli.py ingest evaluation/data/sample_docs/

# 检索
python cli.py query "什么是机器学习？" --top-k 5

# 启动 API 服务（端口 8000）
python cli.py serve

# 启动可视化控制台（端口 7860）
python cli.py ui
```

控制台包含四个工作区：对话、检索实验、评估看板、系统状态。

## CLI 使用

```bash
python cli.py ingest <路径或URL>         # 摄入单个文件或整个目录
python cli.py query "问题"               # 默认返回 Top-5
python cli.py query "问题" --top-k 10    # 返回 Top-10
python cli.py query "问题" --no-reranker # 禁用重排序
python cli.py query "问题" --format json # JSON 输出
python cli.py serve                      # FastAPI 服务（端口 8000）
python cli.py ui                         # Gradio 控制台（端口 7860）
```
## API 接口

启动服务后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

认证：设置环境变量 `HERMES_API_KEY` 后，请求头需携带 `X-API-Key`。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查与索引计数 |
| `/api/v1/stats` | GET | 索引、缓存、指标与脱敏配置 |
| `/api/v1/ingest` | POST | 摄入文件或 URL |
| `/api/v1/query` | POST | 检索问答 |
| `/api/v1/rebuild` | POST | 从文档存储重建向量与 BM25 索引 |

```json
POST /api/v1/query
{
  "query": "什么是机器学习？",
  "top_k": 5,
  "use_reranker": false,
  "generate_answer": false
}
```

## 评估

```bash
# 可复现基准：构建临时索引并输出 JSON 报告
python run_eval.py --output docs/eval_results.json

# 指定变体（baseline / rrf / rerank / cross）
python run_eval.py --variants baseline rrf rerank
```

指标定义：Hit Rate@K（Top-K 命中）、MRR、NDCG@10、Precision@K、Recall@K，以及文档级 Hit Rate@K / Doc MRR。评估数据集位于 `evaluation/data/ground_truth.json`（40 条查询），所有 `relevant_chunk_ids` 均通过自动化测试校验必须真实存在于索引。

## 配置说明

所有参数通过 `config.yaml` 管理，无需修改代码。

| 模块 | 关键参数 | 说明 |
|------|---------|------|
| embedding | model_name, device | 嵌入模型与设备（默认 `BAAI/bge-m3`） |
| chunking | chunk_size, min_chunk_size | 分块大小与最小 token 数（按 token 计，非字符） |
| chromadb | persist_directory, hnsw_M | 向量库与 HNSW 参数 |
| bm25 | persist, fallback_db_path | 稀疏索引持久化 |
| retrieval | dense_top_k, sparse_top_k, rrf_k | 召回与融合参数 |
| reranking | enabled, model_name | 重排序策略（默认关闭） |
| generation | provider, model | LLM 后端（Ollama/OpenAI） |

## 项目结构

```
hermes_rag/
├── cli.py                     # 命令行入口
├── config.yaml                # 全局配置
├── api/                       # FastAPI 服务
├── ui/                        # Gradio 控制台
├── src/
│   └── core/
│       ├── ingestion/         # 文档解析（PDF/DOCX/PPTX/TXT/Web）
│       ├── chunking/          # 层级化分块 + 语义分割
│       ├── indexing/          # ChromaDB + BM25 + DocumentStore
│       ├── retrieval/         # 多路召回 + RRF 融合
│       ├── reranking/         # Cross-Encoder 重排序
│       └── generation/        # LLM 客户端（OpenAI/Ollama）
├── evaluation/                # 评测脚本与数据集
├── docs/                      # 技术文档与评估报告
├── tests/                     # 测试套件
└── .github/workflows/         # CI
```

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 向量数据库 | ChromaDB | 无独立进程、原生 Python、低内存 |
| 嵌入模型 | BAAI/bge-m3 | 1024 维多语言，中英文查询效果稳定 |
| 稀疏检索 | rank_bm25 | 纯 Python，支持 SQLite 持久化 |
| 重排序 | BAAI/bge-reranker-v2-m3 | 多语言跨编码器，默认关闭 |
| 融合算法 | 改进 RRF | 动态权重，k=60 |
| LLM 接口 | OpenAI / Ollama | 可插拔，支持本地部署 |

## License

MIT