# ADR-002: 选择 bge-small-en-v1.5 作为默认 Embedding 模型

## 状态

已采纳

## 背景

需要选择一个 Embedding 模型用于将文本转换为稠密向量。选择标准：

1. 低内存占用（< 500MB）
2. CPU 推理速度快（< 100ms/句）
3. 检索精度高（MTEB 基准）
4. 支持中文（可选）

## 决策

选择 **BAAI/bge-small-en-v1.5** 作为默认模型，同时支持 **BAAI/bge-small-zh** 用于中文场景。

## 理由

### 1. 资源友好

| 模型 | 维度 | 模型大小 | CPU 推理 |
|------|------|----------|----------|
| bge-small-en-v1.5 | 384 | ~130MB | ~50ms |
| bge-base-en-v1.5 | 768 | ~440MB | ~100ms |
| bge-large-en-v1.5 | 1024 | ~1.3GB | ~300ms |
| all-MiniLM-L6-v2 | 384 | ~90MB | ~40ms |

bge-small 在仅 130MB 的体量下，MTEB 检索任务得分 52.3，远高于同尺寸的 all-MiniLM-L6-v2（50.1）。

### 2. CPU 推理性能

在 4核 CPU 上，384 维向量的推理速度约 50ms/句，批处理 32 条约 200ms。满足 < 1.5s 的端到端延迟要求。

### 3. MTEB 基准

bge-small-en-v1.5 在 MTEB Retrieval 任务上排名 small 模型第一，与 base 模型差距仅 1-2 个百分点。

### 4. 中文支持

BGE 系列提供了 bge-small-zh 中文版本，使用相同的维度和接口，可按需切换。

## 替代方案

### all-MiniLM-L6-v2
- 优点：更小（90MB），更快
- 缺点：精度略低，不支持中文

### text2vec-large-chinese
- 优点：中文精度高
- 缺点：体量大（1.3GB），不支持英文

### OpenAI text-embedding-ada-002
- 优点：精度最高
- 缺点：需要 API 调用，有网络延迟和成本

## 后果

- 正面：130MB 模型，CPU 友好，精度/速度平衡最优
- 负面：英文为主，中文精度不如专门的中文模型
- 缓解：config.yaml 中支持切换 `model_name_zh`