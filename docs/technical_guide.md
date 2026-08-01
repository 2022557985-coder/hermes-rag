# Hermes-RAG 技术文档

## 一、架构设计

### 1.1 整体流水线

Hermes-RAG 的检索流水线分为五个阶段：

```
用户查询 → 查询扩展 → 并行多路召回 → RRF 融合 → 可选重排 → 结果返回
```

### 1.2 模块详解

#### 文档摄入（Ingestion）

支持多种文档格式的解析，通过 `ParserFactory` 工厂模式统一路由：

| 格式 | 解析器 | 依赖 |
|------|--------|------|
| PDF | PdfParser | PyMuPDF |
| DOCX | DocxParser | python-docx |
| PPTX | PptxParser | python-pptx |
| TXT/MD | TextParser | 内置 |
| URL | WebParser | BeautifulSoup4 |

#### 层级化分块（Chunking）

采用两阶段分块策略：

1. **标题层级切分**：基于 Markdown 标题（#、##、###）或 PDF 目录树，按章节边界切分
2. **语义分割**：当章节超过 512 tokens 时，使用滑动窗口计算相邻句子余弦相似度，在语义边界处切分

每个 Chunk 绑定元数据：文件名、页码、标题路径、Chunk 索引。

**关键参数**：
- `chunk_size`: 512 tokens
- `chunk_overlap`: 128 tokens
- `semantic_threshold`: 0.65（余弦相似度阈值）
- `min_chunk_size`: 20 tokens（按 token 计数；chunk 文本会前缀完整标题路径）

#### 双索引架构（Indexing）

**稠密向量索引（VectorStore）**：
- 基于 ChromaDB，使用 HNSW 算法
- 嵌入模型：BAAI/bge-m3（1024 维，多语言）
- 索引参数：`ef_construction=100, M=8`

**稀疏 BM25 索引（BM25Index）**：
- 基于 rank_bm25 实现（缺失时使用内置纯 Python 回退）
- 中文分词：jieba；英文分词：nltk
- 支持 SQLite 持久化（`bm25.persist: true`），跨进程保留稀疏索引
- 超过 10万条目时自动迁移到 SQLite 存储

**IndexManager** 协调双索引的并行写入和查询，通过 Chunk ID 关联。启动时执行健康检查：向量维度变化、稀疏索引为空或索引版本升级时，从 DocumentStore 自动重建全部索引。

#### 多路检索（Retrieval）

**查询扩展（QueryExpander）**：
- 同义词扩展：WordNet（英文）+ 内置中文同义词词典（含 80+ 组 ML/IT 领域术语）
- 混合策略：稠密路径使用原始查询，稀疏路径使用扩展查询，避免扩展噪声破坏向量排序
- HyDE（默认关闭）：使用 T5-small 生成假设文档嵌入

**并行召回**：
- DenseRetriever：ChromaDB HNSW 检索 Top-100
- SparseRetriever：BM25 检索 Top-100
- RuleRetriever：正则匹配章节指示词，构建元数据过滤条件

**RRF 融合（RRFFusion）**：
- 改进的 Reciprocal Rank Fusion，k=60
- 动态权重调整：检测产品型号 → BM25 权重 ↑；口语化查询 → 向量权重 ↑

#### 重排序（Reranking）

- 默认关闭：RRF 融合排序为最强变体（`reranking.enabled: false`）
- 可选启发式重排：关键词重叠 + 标题匹配 + 位置信号，原始 RRF 分先归一化到 [0,1] 再加权，避免分数尺度不一致
- 可选 Cross-Encoder：BAAI/bge-reranker-v2-m3，对 Top-50 候选逐对打分
- 超时保护：超时降级为 RRF 结果；惰性加载：仅在检索时加载模型

#### LLM 生成（Generation）

可插拔设计，支持两种后端：
- **Ollama**：本地部署（推荐 qwen2.5:7b）
- **OpenAI**：兼容 OpenAI API 的服务

### 1.3 资源优化

- **内存管理**：BM25 在超过 10万条目时自动迁移到 SQLite，释放内存
- **模型复用**：Embedding 模型和 Reranker 模型在会话期间保持加载
- **查询缓存**：语义相似度缓存（阈值 0.95），减少重复计算
- **GC 清理**：每次检索后执行 `gc.collect()`

## 二、关键算法

### 2.1 改进 RRF 融合

标准 RRF 公式：
```
RRF(d) = Σ(1 / (k + rank_i(d)))
```

改进版引入动态权重：
```
RRF(d) = w_dense × (1/(k + rank_dense(d))) + w_sparse × (1/(k + rank_sparse(d)))
```

权重根据查询特征动态调整：
- 包含产品型号/编号 → w_sparse = 0.7, w_dense = 0.3
- 自然语言口语化 → w_dense = 0.8, w_sparse = 0.2
- 默认 → w_dense = 0.5, w_sparse = 0.5

### 2.2 语义分块

使用 Sentence-BERT 计算相邻句子的余弦相似度，在相似度低于阈值的位置切分。

```
句子1 → [embedding1]
句子2 → [embedding2]
相似度 = cosine(embedding1, embedding2)
如果 相似度 < threshold → 切分
```

### 2.3 语义缓存

缓存查询的嵌入向量，新查询与缓存中查询的余弦相似度 > 0.95 时命中：
```
similarity = cosine(embedding(query_new), embedding(query_cached))
if similarity > 0.95 → 返回缓存结果
```

## 三、配置管理

使用 Pydantic Settings + YAML 实现类型安全的配置管理：

```python
class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32
```

配置文件 `config.yaml` 支持环境变量替换（`${VAR_NAME}`），所有参数均可不修改代码地进行调整。

## 四、性能基准

在 8GB RAM + 4核CPU 环境下：

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 索引构建 | < 100 docs/s | 含嵌入生成 |
| 单次查询延迟 | < 1.5s | 含重排序 |
| 内存占用 | < 3GB | 含所有模型 |
| Hit Rate@5 | 100.0%（40 条查询） | 默认 RRF 变体，`run_eval.py` 实测 |

## 五、扩展指南

### 5.1 添加新的文档格式

1. 在 `src/core/ingestion/` 下创建新的解析器
2. 实现 `parse(file_path) -> dict` 方法
3. 在 `ParserFactory` 中注册

### 5.2 添加新的 Embedding 模型

1. 修改 `config.yaml` 中的 `embedding.model_name`
2. 模型需支持 sentence-transformers 接口

### 5.3 添加新的 LLM 后端

1. 在 `LLMClient` 中添加新的 provider 分支
2. 实现 `generate(query, context) -> str` 方法