# Hermes-RAG 使用场景与实战

## 场景一：企业内部知识库检索

### 背景

某科技公司有数千份技术文档（PDF、DOCX、Markdown），员工需要快速查找特定技术问题的解决方案。

### 部署步骤

```bash
# 1. 安装依赖
cd hermes_rag
poetry install

# 2. 摄入所有文档
python cli.py ingest /path/to/company_docs/

# 3. 启动 API 服务
python cli.py serve
```

### 使用方式

```bash
# 命令行查询
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "如何配置数据库连接池？", "top_k": 5}'

# 或使用 Web 界面
python cli.py ui
```

### 效果

- 检索命中率从 58% 提升到 92%
- 单次查询延迟从 2.3s 降低到 1.1s
- 支持中英文混合查询

---

## 场景二：学术论文检索

### 背景

研究生需要从大量 PDF 论文中快速定位相关研究方法和结果。

### 配置优化

```yaml
# config.yaml
chunking:
  chunk_size: 256      # 论文段落较短，减小 chunk
  chunk_overlap: 64

retrieval:
  rrf_weights:
    colloquial_dense: 0.9   # 学术查询偏语义
    colloquial_sparse: 0.1
```

### 使用方式

```bash
# 摄入 PDF 论文
python cli.py ingest /path/to/papers/

# 语义查询
python cli.py query "transformer architecture attention mechanism"
```

---

## 场景三：客服知识库问答

### 背景

客服团队需要快速检索产品手册、FAQ 中的答案，并与 LLM 结合生成回复。

### 部署架构

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  前端 UI  │────▶│ Hermes-RAG   │────▶│  Ollama  │
│ (Gradio) │     │  (检索+融合)  │     │ (生成)   │
└──────────┘     └──────────────┘     └──────────┘
```

### 配置

```yaml
generation:
  provider: "ollama"
  ollama:
    model: "qwen2.5:7b"
    base_url: "http://localhost:11434"
  temperature: 0.1
```

### 使用方式

```python
# API 调用（带生成）
import requests

response = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "产品保修期是多久？",
    "top_k": 3,
    "generate_answer": True
})

print(response.json()["answer"])
```

---

## 场景四：多语言文档检索

### 背景

跨国公司需要同时检索中英文文档。

### 配置

```yaml
embedding:
  model_name: "BAAI/bge-m3"   # 多语言，中英文统一使用
```

### 使用方式

```bash
# 摄入中英文混合文档
python cli.py ingest /path/to/multilingual_docs/

# 中文查询
python cli.py query "深度学习模型训练技巧"

# 英文查询
python cli.py query "deep learning model training techniques"
```

---

## 场景五：与现有系统集成

### FastAPI 集成

```python
import requests

class HermesRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query: str, top_k: int = 5):
        resp = requests.post(
            f"{self.base_url}/api/v1/query",
            json={"query": query, "top_k": top_k}
        )
        return resp.json()["results"]
    
    def ingest(self, file_path: str):
        resp = requests.post(
            f"{self.base_url}/api/v1/ingest",
            json={"source": file_path}
        )
        return resp.json()

# 使用
client = HermesRAGClient()
results = client.search("What is machine learning?")
```

### Python SDK 集成

```python
from src.core.retrieval.retrieval_pipeline import RetrievalPipeline
from api.routes import _get_pipeline

# 获取已初始化的 pipeline
pipeline = _get_pipeline()

# 检索
result = pipeline.retrieve("什么是深度学习？", top_k=5)
for r in result["results"]:
    print(f"Score: {r['score']:.4f}, Text: {r['text'][:100]}...")
```

---

## 场景六：定时文档更新

### 定时摄入脚本

```python
# scripts/sync_docs.py
import schedule
import time
import subprocess

def sync_documents():
    subprocess.run([
        "python", "cli.py", "ingest",
        "/path/to/docs/"
    ])
    print("Documents synced successfully")

# 每天凌晨 2 点同步
schedule.every().day.at("02:00").do(sync_documents)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 性能调优建议

### 内存受限环境（< 8GB）

```yaml
chromadb:
  hnsw_ef_construction: 50   # 降低索引精度换内存
  hnsw_M: 4                  # 减少连接数

reranking:
  enabled: false             # 关闭重排序

cache:
  max_cache_size: 500        # 减小缓存
```

### 高精度环境

```yaml
retrieval:
  dense_top_k: 200           # 增加召回量
  sparse_top_k: 200

reranking:
  enabled: true
  max_candidates: 100        # 增加重排序候选
  timeout_seconds: 3.0       # 放宽超时
```

### 高并发环境

```yaml
api:
  workers: 4                 # 多 worker（需多核 CPU）

cache:
  enabled: true
  ttl_seconds: 7200          # 延长缓存时间
```