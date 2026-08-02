"""Gradio web interface for Hermes-RAG.

Provides four workspaces in one app:
- Chat: multi-turn retrieval-augmented conversation with feedback
- Retrieval Lab: single-query result inspection with configurable knobs
- Evaluation Dashboard: canonical benchmark report from docs/eval_results.json
- System Status: live index, cache, metrics and configuration summary
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr  # noqa: E402

# Global stop signal for in-progress LLM generation
_stop_event = threading.Event()

_EVAL_PATH = _project_root / "docs" / "eval_results.json"

DARK_MODE_JS = """
function toggleDark() {
    document.body.classList.toggle('hermes-dark');
    const isDark = document.body.classList.contains('hermes-dark');
    localStorage.setItem('hermes-dark-mode', isDark ? '1' : '0');
}
window.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('hermes-dark-mode') === '1') {
        document.body.classList.add('hermes-dark');
    }
});
"""

CUSTOM_CSS = """
:root {
    --bg: #f4f6f8;
    --surface: #ffffff;
    --surface-2: #eef2f4;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #0f766e;
    --accent-2: #b45309;
    --border: #d7dee3;
}
body, .gradio-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.hermes-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 4px 6px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.hermes-header .logo {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: linear-gradient(135deg, #0f766e, #0d9488);
    color: #fff;
    font-size: 20px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.hermes-header h1 {
    margin: 0;
    font-size: 20px;
    line-height: 1.2;
}
.hermes-header p {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
}
.hermes-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 10px 0;
}
.hermes-metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
}
.hermes-metric .label {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
}
.hermes-metric .value {
    font-size: 20px;
    font-weight: 650;
    color: var(--text);
}
.hermes-metric .sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
}
.hermes-bars {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
}
.hermes-bar-row {
    display: grid;
    grid-template-columns: 110px 1fr 56px;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}
.hermes-bar-track {
    height: 12px;
    background: var(--surface-2);
    border-radius: 6px;
    overflow: hidden;
}
.hermes-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #0f766e, #2dd4bf);
}
.hermes-table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
}
body.hermes-dark {
    --bg: #16181d;
    --surface: #20242b;
    --surface-2: #2a2f38;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #2dd4bf;
    --accent-2: #f59e0b;
    --border: #343a44;
}
body.hermes-dark, body.hermes-dark .gradio-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
body.hermes-dark .prose,
body.hermes-dark .markdown,
body.hermes-dark .chatbot {
    color: var(--text) !important;
}
body.hermes-dark input,
body.hermes-dark textarea,
body.hermes-dark select,
body.hermes-dark .dataframe {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
body.hermes-dark button {
    color: var(--text) !important;
}
"""


def _build_llm_client():
    """Build LLM client from config."""
    from src.config import get_config
    from src.core.generation.llm_client import LLMClient

    cfg = get_config()
    provider = cfg.generation.provider
    if provider == "openai":
        return LLMClient(
            provider="openai",
            model=cfg.generation.openai.model,
            base_url=cfg.generation.openai.base_url,
            api_key=cfg.generation.openai.api_key,
            temperature=cfg.generation.temperature,
            max_context_tokens=cfg.generation.max_context_tokens,
        )
    return LLMClient(
        provider="ollama",
        model=cfg.generation.ollama.model,
        base_url=cfg.generation.ollama.base_url,
        temperature=cfg.generation.temperature,
        max_context_tokens=cfg.generation.max_context_tokens,
    )


def _check_pipeline_available():
    """Check if the pipeline is available."""
    try:
        from api.routes import _get_pipeline
        _get_pipeline()
        return True, ""
    except Exception as e:
        return False, str(e)


def _clear_cache():
    """Clear the query cache."""
    try:
        from api.routes import _get_pipeline
        pipeline = _get_pipeline()
        if hasattr(pipeline, "cache") and pipeline.cache is not None:
            pipeline.cache.clear()
            return "缓存已清除", _get_metrics_html(), _system_status_html()
        return "缓存对象不可用", _get_metrics_html(), _system_status_html()
    except Exception as e:
        return f"清除缓存失败: {e}", _get_metrics_html(), _system_status_html()

def _reload_knowledge_base():
    """Rebuild the index from the configured document directory."""
    try:
        from src.config import get_config
        from src.core.chunking.hierarchical_chunker import HierarchicalChunker
        from src.core.ingestion.parser_factory import ParserFactory
        from src.core.pipeline_factory import build_pipeline
        from src.utils.security import (
            validate_file_extension,
            validate_file_path,
            validate_file_size,
        )
        from src.utils.timer import Stopwatch

        cfg = get_config()
        doc_dir = cfg.auto_ingest.doc_dir
        if not Path(doc_dir).exists():
            return f"目录不存在: {doc_dir}", _get_metrics_html(), _system_status_html()

        sw = Stopwatch()
        pipeline = build_pipeline(config=cfg)
        index_manager = pipeline.index_manager
        index_manager.clear()

        chunker = HierarchicalChunker(
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            semantic_threshold=cfg.chunking.semantic_threshold,
            min_chunk_size=cfg.chunking.min_chunk_size,
            max_section_size=cfg.chunking.max_section_size,
            embedding_model=cfg.embedding.model_name,
            embedding_device=cfg.embedding.device,
        )

        files = []
        for root, _, filenames in os.walk(doc_dir):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in (
                    ".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".markdown",
                    ".rst", ".csv", ".json", ".html", ".htm", ".log",
                ):
                    files.append(os.path.join(root, fname))

        if not files:
            return f"目录 {doc_dir} 中无支持的文件", _get_metrics_html(), _system_status_html()

        total_chunks = 0
        skipped = 0
        for file_path in files:
            try:
                validate_file_path(file_path)
                validate_file_extension(file_path)
                validate_file_size(file_path)
                parsed = ParserFactory.get_parser(file_path).parse(file_path)
                chunks = chunker.chunk(
                    text=parsed.get("text", ""),
                    source_name=os.path.basename(file_path),
                    headings=parsed.get("metadata", {}).get("headings"),
                )
                if chunks:
                    index_manager.ingest_chunks(chunks)
                    total_chunks += len(chunks)
            except Exception:
                skipped += 1

        elapsed = sw.elapsed()
        message = f"已重载 {len(files)} 个文件 ({total_chunks} chunks, {elapsed:.2f}s)"
        if skipped:
            message += f", 跳过 {skipped} 个"
        return message, _get_metrics_html(), _system_status_html()
    except Exception as e:
        return f"重载失败: {e}", _get_metrics_html(), _system_status_html()


def _get_metrics_html():
    """Generate metrics display HTML."""
    try:
        from src.utils.metrics import get_metrics
        report = get_metrics().get_full_report()
        lat = report["latency"]
        cache = report["cache"]
        return (
            "<div style='font-size:13px;line-height:1.7'>"
            f"<b>总查询:</b> {report['total_queries']} &nbsp; <b>QPS:</b> {report['qps']}<br>"
            f"<b>缓存命中率:</b> {cache['hit_rate']:.1%} ({cache['hits']} 命中 / {cache['misses']} 未命中)<br>"
            f"<b>平均延迟:</b> {lat['avg_ms']}ms &nbsp; <b>P50:</b> {lat['p50_ms']}ms &nbsp; <b>P95:</b> {lat['p95_ms']}ms<br>"
            f"<b>失败率:</b> {report['failure_rate']:.1%}"
            "</div>"
        )
    except Exception:
        return "<div style='font-size:13px;color:#999'>指标未初始化</div>"


def _format_results_markdown(results, recall_paths, from_cache, timing):
    """Format retrieval results as rich markdown with source highlighting."""
    if from_cache:
        source_label = "缓存命中"
    elif recall_paths:
        if "dense" in recall_paths and "sparse" in recall_paths:
            source_label = "稠密 + 稀疏混合检索"
        elif "dense" in recall_paths:
            source_label = "稠密语义检索"
        elif "sparse" in recall_paths:
            source_label = "稀疏关键词检索"
        else:
            source_label = "检索"
    else:
        source_label = "检索"

    if not results:
        return f"**{source_label}** | 未找到相关结果"

    lines = [f"**{source_label}** | 耗时: {timing:.3f}s | 返回 {len(results)} 条结果\n"]
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("metadata", {}).get("source", "unknown")
        heading = r.get("metadata", {}).get("heading_path", "")
        is_neighbor = r.get("_is_neighbor", False)
        badge = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
        neighbor_tag = " 📎相邻" if is_neighbor else ""
        header = f"### [{i}] {badge} Score: {score:.4f}{neighbor_tag}"
        if heading:
            header += f" | 📂 {heading}"
        header += f"\n> 📄 来源: `{source}`"
        lines.append(f"{header}\n\n{r.get('text', '')}\n\n---\n")
    return "\n".join(lines)


def _build_chatbot_message(role, content):
    """Build a chatbot message dict."""
    return {"role": role, "content": content}


def _record_feedback(query, results, rating):
    """Record user feedback for quality improvement."""
    try:
        feedback_dir = _project_root / "data" / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "rating": rating,
            "top_chunk_ids": [r.get("chunk_id", "") for r in (results or [])[:3]],
            "top_scores": [r.get("score", 0) for r in (results or [])[:3]],
        }
        with open(feedback_dir / "feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def query_hermes(query, top_k, use_reranker, generate_answer, chat_history, request):
    """Handle chat query with retrieval and optional streaming generation."""
    global _stop_event
    _stop_event.clear()

    if not query or not query.strip():
        return chat_history, chat_history, "", "就绪"

    try:
        from api.routes import _get_pipeline

        pipeline = _get_pipeline()
        result = pipeline.retrieve(query=query, top_k=top_k, use_reranker=use_reranker)
        recall_paths = result.get("recall_paths", [])
        from_cache = result.get("from_cache", False)
        timing = result.get("timing", {}).get("total", 0)
        results = result.get("results", [])

        results_md = _format_results_markdown(results, recall_paths, from_cache, timing)
        timing_info = f"检索: {timing:.3f}s | {'缓存' if from_cache else '实时'}"

        if chat_history is None:
            chat_history = []
        chat_history.append(_build_chatbot_message("user", query))

        if not generate_answer or not results:
            chat_history.append(_build_chatbot_message("assistant", results_md))
            yield chat_history, chat_history, timing_info, "检索完成"
            return

        chat_history.append(
            _build_chatbot_message(
                "assistant", f"{results_md}\n\n---\n\n🤖 **AI 回答（生成中...）**\n\n"
            )
        )
        yield chat_history, chat_history, f"{timing_info} | 生成中...", "生成中"

        try:
            llm = _build_llm_client()
            prompt = llm._build_prompt(query, results)
            num_predict = min(llm.max_context_tokens, 1024)

            import requests as req
            response = req.post(
                f"{llm.base_url}/api/generate",
                json={
                    "model": llm.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": llm.temperature, "num_predict": num_predict},
                },
                timeout=300,
                stream=True,
            )
            response.raise_for_status()

            answer_parts = []
            gen_start = time.time()
            last_yield_time = gen_start
            for line in response.iter_lines():
                if _stop_event.is_set():
                    break
                if line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "response" in data:
                        answer_parts.append(data["response"])
                        now = time.time()
                        if now - last_yield_time >= 0.3:
                            current_answer = "".join(answer_parts)
                            chat_history[-1] = _build_chatbot_message(
                                "assistant", f"{results_md}\n\n---\n\n🤖 **AI 回答**\n\n{current_answer}"
                            )
                            yield chat_history, chat_history, f"{timing_info} | 生成中... ({now - gen_start:.1f}s)", "生成中"
                            last_yield_time = now
                    if data.get("done", False):
                        break

            if _stop_event.is_set():
                answer_parts.append("\n\n[生成已停止]")
                status = "已停止"
            else:
                status = "完成"
            answer = "".join(answer_parts)
            gen_time = time.time() - gen_start
            chat_history[-1] = _build_chatbot_message(
                "assistant", f"{results_md}\n\n---\n\n🤖 **AI 回答** ({gen_time:.1f}s)\n\n{answer}"
            )
            yield chat_history, chat_history, f"{timing_info} | 生成: {gen_time:.1f}s", status
        except req.exceptions.ConnectionError:
            chat_history[-1] = _build_chatbot_message(
                "assistant", f"{results_md}\n\n---\n\n❌ **LLM 服务连接失败**"
            )
            yield chat_history, chat_history, timing_info, "连接失败"
        except Exception as e:
            chat_history[-1] = _build_chatbot_message(
                "assistant", f"{results_md}\n\n---\n\n❌ **LLM 生成失败:** {e}"
            )
            yield chat_history, chat_history, timing_info, "生成失败"
    except Exception as e:
        if chat_history is None:
            chat_history = []
        chat_history.append(_build_chatbot_message("user", query))
        chat_history.append(_build_chatbot_message("assistant", f"❌ 查询失败: {e}"))
        yield chat_history, chat_history, "", "出错"


def _stop_generation():
    """Signal to stop in-progress LLM generation."""
    global _stop_event
    _stop_event.set()
    return "已停止"


def _feedback_thumbs_up(query, chat_history):
    """Record positive feedback."""
    if chat_history and len(chat_history) >= 2:
        _record_feedback(query, [], "up")
    return "已记录 👍"


def _feedback_thumbs_down(query, chat_history):
    """Record negative feedback."""
    if chat_history and len(chat_history) >= 2:
        _record_feedback(query, [], "down")
    return "已记录 👎"


def _clear_chat():
    """Clear chat history."""
    return [], [], "就绪"


def _load_eval_report():
    """Load the canonical benchmark report as a dict (or None)."""
    try:
        with open(_EVAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pct(value):
    try:
        return value * 100
    except TypeError:
        return 0.0


def _variant_rows(report):
    """Build table rows for retrieval variants."""
    rows = []
    for name, m in (report or {}).get("variants", {}).items():
        rows.append(
            [
                name,
                f"{_pct(m.get('hit_rate@1', 0)):.1f}%",
                f"{_pct(m.get('hit_rate@3', 0)):.1f}%",
                f"{_pct(m.get('hit_rate@5', 0)):.1f}%",
                f"{_pct(m.get('doc_hit_rate@5', 0)):.1f}%",
                f"{m.get('mrr', 0):.4f}",
                f"{m.get('ndcg@10', 0):.4f}",
                f"{m.get('avg_latency', 0) * 1000:.0f}ms",
            ]
        )
    return rows


def _difficulty_rows(report):
    """Build table rows for difficulty breakdown."""
    rows = []
    for diff, m in (report or {}).get("difficulty_breakdown", {}).items():
        rows.append(
            [
                diff,
                m.get("queries", 0),
                f"{_pct(m.get('hit_rate@1', 0)):.1f}%",
                f"{_pct(m.get('hit_rate@3', 0)):.1f}%",
                f"{_pct(m.get('hit_rate@5', 0)):.1f}%",
                f"{_pct(m.get('doc_hit_rate@5', 0)):.1f}%",
                f"{m.get('mrr', 0):.4f}",
                f"{m.get('ndcg@10', 0):.4f}",
            ]
        )
    return rows


def _failed_rows(report):
    """Build table rows for failed queries."""
    rows = []
    for item in (report or {}).get("failed_queries", [])[:20]:
        rows.append([item.get("difficulty", ""), item.get("query", ""), ", ".join(item.get("expected", []))])
    return rows

def _eval_overview_html(report):
    """Render evaluation summary cards and H@K bars."""
    if not report:
        return "<div style='color:#6b7280'>评估结果尚未生成</div>"
    meta = report.get("meta", {})
    variants = report.get("variants", {})
    primary_name = "rrf_no_rerank" if "rrf_no_rerank" in variants else next(iter(variants), None)
    primary = variants.get(primary_name, {})
    difficulty = report.get("difficulty_breakdown", {})

    cards = []
    cards.append(f"<div class='hermes-metric'><div class='label'>评测日期</div><div class='value'>{meta.get('date', '-')}</div><div class='sub'>Python {meta.get('python', '-')}</div></div>")
    cards.append(f"<div class='hermes-metric'><div class='label'>查询数</div><div class='value'>{meta.get('dataset_size', 0)}</div><div class='sub'>索引 {meta.get('chunks', 0)} chunks</div></div>")
    cards.append(f"<div class='hermes-metric'><div class='label'>主变体 H@1</div><div class='value'>{_pct(primary.get('hit_rate@1', 0)):.1f}%</div><div class='sub'>{primary_name}</div></div>")
    cards.append(f"<div class='hermes-metric'><div class='label'>主变体 H@5</div><div class='value'>{_pct(primary.get('hit_rate@5', 0)):.1f}%</div><div class='sub'>MRR {primary.get('mrr', 0):.4f}</div></div>")
    cards.append(f"<div class='hermes-metric'><div class='label'>文档级 H@5</div><div class='value'>{_pct(primary.get('doc_hit_rate@5', 0)):.1f}%</div><div class='sub'>Doc MRR {primary.get('doc_mrr', 0):.4f}</div></div>")
    cards.append(f"<div class='hermes-metric'><div class='label'>失败查询</div><div class='value'>{len(report.get('failed_queries', []))}</div><div class='sub'>平均延迟 {primary.get('avg_latency', 0) * 1000:.0f}ms</div></div>")

    bar_rows = []
    for diff, m in difficulty.items():
        h5 = _pct(m.get("hit_rate@5", 0))
        bar_rows.append(
            f"<div class='hermes-bar-row'><span>{diff}</span><div class='hermes-bar-track'>"
            f"<div class='hermes-bar-fill' style='width:{h5:.1f}%'></div></div>"
            f"<span>{h5:.1f}%</span></div>"
        )
    bars = f"<div class='hermes-bars'>{''.join(bar_rows)}</div>" if bar_rows else ""

    html = (
        "<div class='hermes-metric-grid'>" + "".join(cards) + "</div>"
        "<div class='hermes-table-wrap'><div style='font-size:13px;font-weight:650;margin-bottom:4px'>Hit Rate@5 by difficulty</div>" + bars + "</div>"
    )
    return html


def _redact_config(data):
    """Remove secrets from a config dict."""
    if isinstance(data, dict):
        return {k: _redact_config(v) if isinstance(v, dict) else ("***" if k.lower() in ("api_key", "password", "token", "secret") else v) for k, v in data.items()}
    return data


def _system_status_json():
    """Collect live system status as a JSON-serializable dict."""
    try:
        from api.routes import _get_pipeline
        from src.config import get_config
        from src.utils.metrics import get_metrics

        pipeline = _get_pipeline()
        index = {}
        try:
            index = pipeline.index_manager.get_stats()
        except Exception as e:
            index = {"error": str(e)}
        cache = {"enabled": pipeline.cache is not None}
        if pipeline.cache is not None:
            try:
                cache.update(pipeline.cache.get_stats())
            except Exception as e:
                cache["error"] = str(e)
        metrics = {}
        try:
            metrics = get_metrics().get_full_report()
        except Exception as e:
            metrics = {"error": str(e)}
        cfg = _redact_config(get_config().to_dict())
        return {
            "status": "healthy",
            "index": index,
            "cache": cache,
            "metrics": metrics,
            "config": cfg,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _system_status_html():
    """Render live status as metric cards."""
    try:
        data = _system_status_json()
        index = data.get("index", {})
        vector = index.get("vector", {}) if isinstance(index, dict) else {}
        bm25 = index.get("bm25", {}) if isinstance(index, dict) else {}
        doc = index.get("document_store", {}) if isinstance(index, dict) else {}
        cache = data.get("cache", {})
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        generation = config.get("generation", {})

        cards = []
        cards.append(f"<div class='hermes-metric'><div class='label'>向量索引</div><div class='value'>{vector.get('total_chunks', 'N/A')}</div><div class='sub'>{config.get('embedding', {}).get('model_name', '-')}</div></div>")
        cards.append(f"<div class='hermes-metric'><div class='label'>BM25 索引</div><div class='value'>{bm25.get('total_chunks', 'N/A')}</div><div class='sub'>{bm25.get('mode', '-')} 模式</div></div>")
        cards.append(f"<div class='hermes-metric'><div class='label'>文档存储</div><div class='value'>{doc.get('total_chunks', 'N/A')}</div><div class='sub'>SQLite</div></div>")
        cards.append(f"<div class='hermes-metric'><div class='label'>缓存</div><div class='value'>{cache.get('size', 'N/A')}</div><div class='sub'>命中率 {_pct(cache.get('hit_rate', 0)):.1f}%</div></div>")
        cards.append(f"<div class='hermes-metric'><div class='label'>生成服务</div><div class='value'>{generation.get('provider', '-')}</div><div class='sub'>{generation.get('ollama', {}).get('model', generation.get('openai', {}).get('model', '-'))}</div></div>")

        lat = metrics.get("latency", {}) if isinstance(metrics, dict) else {}
        cards.append(f"<div class='hermes-metric'><div class='label'>平均延迟</div><div class='value'>{lat.get('avg_ms', 'N/A')}ms</div><div class='sub'>总查询 {metrics.get('total_queries', 0)}</div></div>")
        return "<div class='hermes-metric-grid'>" + "".join(cards) + "</div>"
    except Exception as e:
        return f"<div style='color:#b91c1c'>状态获取失败: {e}</div>"


def _example_queries():
    """Return a small set of ready-to-test queries."""
    try:
        from evaluation.dataset import EvaluationDataset
        queries = EvaluationDataset().get_queries()
        if queries:
            return queries[:12]
    except Exception:
        pass
    return ["什么是机器学习？", "什么是神经网络？", "分类和回归有什么区别？"]


def _run_lab_query(query, top_k, use_reranker, clear_cache):
    """Run a single retrieval experiment and return table + detail markdown."""
    if not query or not query.strip():
        return [], "", "请输入查询", "待执行"
    try:
        from api.routes import _get_pipeline
        pipeline = _get_pipeline()
        if clear_cache and pipeline.cache is not None:
            pipeline.cache.clear()
        result = pipeline.retrieve(query=query, top_k=top_k, use_reranker=use_reranker)
        results = result.get("results", [])
        timing = result.get("timing", {}).get("total", 0)
        recall_paths = result.get("recall_paths", [])
        from_cache = result.get("from_cache", False)

        rows = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            rows.append(
                [
                    i,
                    r.get("chunk_id", ""),
                    f"{r.get('score', 0):.4f}",
                    meta.get("source", ""),
                    meta.get("heading_path", ""),
                    (r.get("text", "") or "")[:120],
                ]
            )
        detail = _format_results_markdown(results, recall_paths, from_cache, timing)
        timing_line = f"检索: {timing:.3f}s | {'缓存' if from_cache else '实时'} | 路径: {', '.join(recall_paths) if recall_paths else '-'}"
        return rows, detail, timing_line, "完成"
    except Exception as e:
        return [], "", f"查询失败: {e}", "出错"


def _fill_example(example):
    """Return the selected example query."""
    return example or ""

def create_ui():
    """Create the Hermes-RAG web interface."""
    pipeline_ok, pipeline_error = _check_pipeline_available()

    with gr.Blocks(title="Hermes-RAG 控制台") as demo:
        chat_history_state = gr.State([])

        gr.HTML(
            """
            <div class='hermes-header'>
                <div class='logo'>H</div>
                <div>
                    <h1>Hermes-RAG</h1>
                    <p>本地化检索增强问答 · 检索实验 · 评估看板 · 系统状态</p>
                </div>
            </div>
            """
        )

        if not pipeline_ok:
            gr.Markdown(f"⚠️ **管道加载失败**\n\n`{pipeline_error}`")

        with gr.Tabs():
            with gr.Tab("对话"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=520,
                            buttons=["copy"],
                        )
                        with gr.Row():
                            thumbs_up_btn = gr.Button("👍", size="sm", scale=0, min_width=44)
                            thumbs_down_btn = gr.Button("👎", size="sm", scale=0, min_width=44)
                            feedback_msg = gr.Textbox(label="", interactive=False, visible=False, scale=0)
                            clear_chat_btn = gr.Button("清空对话", size="sm", scale=1)

                        query_input = gr.Textbox(
                            label="输入问题",
                            placeholder="输入问题...",
                            lines=2,
                        )
                        with gr.Row():
                            query_btn = gr.Button("检索", variant="primary", scale=2)
                            stop_btn = gr.Button("停止生成", variant="stop", visible=False, scale=1)
                        with gr.Row():
                            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="返回结果数")
                            reranker_toggle = gr.Checkbox(value=False, label="重排序")
                            gen_answer_toggle = gr.Checkbox(value=False, label="AI 回答")
                        with gr.Row():
                            clear_cache_btn = gr.Button("清除缓存", size="sm")
                            reload_kb_btn = gr.Button("重载知识库", size="sm")
                            dark_mode_btn = gr.Button("暗色模式", size="sm")

                    with gr.Column(scale=1):
                        progress_output = gr.Textbox(label="状态", interactive=False, value="就绪")
                        timing_output = gr.Textbox(label="性能", interactive=False)
                        metrics_display = gr.HTML(value=_get_metrics_html(), every=3)

            with gr.Tab("检索实验"):
                with gr.Row():
                    with gr.Column(scale=2):
                        lab_query = gr.Textbox(label="查询", lines=3, placeholder="输入实验查询...")
                        with gr.Row():
                            lab_top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                            lab_reranker = gr.Checkbox(value=False, label="重排序")
                            lab_clear_cache = gr.Checkbox(value=False, label="查询前清空缓存")
                        with gr.Row():
                            lab_run_btn = gr.Button("运行检索", variant="primary")
                            lab_example_dd = gr.Dropdown(choices=_example_queries(), label="示例问题")
                            lab_fill_btn = gr.Button("填入")
                        lab_status = gr.Textbox(label="状态", interactive=False, value="待执行")
                        lab_timing = gr.Textbox(label="性能", interactive=False)
                    with gr.Column(scale=3):
                        lab_table = gr.Dataframe(
                            headers=["#", "chunk_id", "score", "source", "heading", "text 预览"],
                            datatype=["number", "str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )
                        lab_detail = gr.Markdown("")

            with gr.Tab("评估看板"):
                with gr.Row():
                    eval_refresh_btn = gr.Button("刷新评估", variant="secondary")
                    eval_meta = gr.Markdown("")
                eval_overview = gr.HTML(value=_eval_overview_html(_load_eval_report()))
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 检索变体")
                        eval_variants = gr.Dataframe(
                            headers=["变体", "H@1", "H@3", "H@5", "文档H@5", "MRR", "NDCG@10", "延迟"],
                            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                            interactive=False,
                            value=_variant_rows(_load_eval_report()),
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### 难度拆解")
                        eval_difficulty = gr.Dataframe(
                            headers=["难度", "查询数", "H@1", "H@3", "H@5", "文档H@5", "MRR", "NDCG@10"],
                            datatype=["str", "number", "str", "str", "str", "str", "str", "str"],
                            interactive=False,
                            value=_difficulty_rows(_load_eval_report()),
                        )
                gr.Markdown("#### 失败查询")
                eval_failed = gr.Dataframe(
                    headers=["难度", "查询", "期望 chunk"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    value=_failed_rows(_load_eval_report()),
                )

            with gr.Tab("系统状态"):
                with gr.Row():
                    status_refresh_btn = gr.Button("刷新状态", variant="secondary")
                status_html = gr.HTML(value=_system_status_html(), every=5)
                status_json = gr.JSON(value=_system_status_json(), every=5)

        def _handle_query(query, top_k, use_reranker, gen_answer, history):
            yield from query_hermes(
                query, top_k, use_reranker, gen_answer, history, None
            )

        query_input.submit(
            fn=_handle_query,
            inputs=[query_input, top_k_slider, reranker_toggle, gen_answer_toggle, chat_history_state],
            outputs=[chatbot, chat_history_state, timing_output, progress_output],
        ).then(
            fn=_get_metrics_html, inputs=None, outputs=[metrics_display], queue=False,
        ).then(
            fn=lambda: "", inputs=None, outputs=[query_input], queue=False,
        )

        query_btn.click(
            fn=_handle_query,
            inputs=[query_input, top_k_slider, reranker_toggle, gen_answer_toggle, chat_history_state],
            outputs=[chatbot, chat_history_state, timing_output, progress_output],
        ).then(
            fn=_get_metrics_html, inputs=None, outputs=[metrics_display], queue=False,
        ).then(
            fn=lambda: "", inputs=None, outputs=[query_input], queue=False,
        )

        stop_btn.click(fn=_stop_generation, inputs=None, outputs=[progress_output], queue=False)
        clear_chat_btn.click(fn=_clear_chat, inputs=None, outputs=[chatbot, chat_history_state, progress_output], queue=False)

        thumbs_up_btn.click(
            fn=_feedback_thumbs_up,
            inputs=[query_input, chat_history_state],
            outputs=[feedback_msg],
            queue=False,
        )
        thumbs_down_btn.click(
            fn=_feedback_thumbs_down,
            inputs=[query_input, chat_history_state],
            outputs=[feedback_msg],
            queue=False,
        )

        clear_cache_btn.click(
            fn=_clear_cache,
            inputs=None,
            outputs=[progress_output, metrics_display, status_html],
            queue=False,
        )
        reload_kb_btn.click(
            fn=_reload_knowledge_base,
            inputs=None,
            outputs=[progress_output, metrics_display, status_html],
            queue=False,
        )
        gen_answer_toggle.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[gen_answer_toggle],
            outputs=[stop_btn],
            queue=False,
        )
        dark_mode_btn.click(fn=None, inputs=None, outputs=None, js="toggleDark()")

        lab_run_btn.click(
            fn=_run_lab_query,
            inputs=[lab_query, lab_top_k, lab_reranker, lab_clear_cache],
            outputs=[lab_table, lab_detail, lab_timing, lab_status],
            queue=False,
        )
        lab_query.submit(
            fn=_run_lab_query,
            inputs=[lab_query, lab_top_k, lab_reranker, lab_clear_cache],
            outputs=[lab_table, lab_detail, lab_timing, lab_status],
            queue=False,
        )
        lab_fill_btn.click(fn=_fill_example, inputs=[lab_example_dd], outputs=[lab_query], queue=False)

        def _refresh_eval():
            report = _load_eval_report()
            meta = report.get("meta", {}) if report else {}
            meta_line = f"评测时间: {meta.get('date', '-')} | 查询数: {meta.get('dataset_size', 0)} | 索引: {meta.get('chunks', 0)} chunks | 嵌入模型: {meta.get('embedding_model', '-')}"
            return meta_line, _eval_overview_html(report), _variant_rows(report), _difficulty_rows(report), _failed_rows(report)

        eval_refresh_btn.click(
            fn=_refresh_eval,
            inputs=None,
            outputs=[eval_meta, eval_overview, eval_variants, eval_difficulty, eval_failed],
            queue=False,
        )

        def _refresh_status():
            return _system_status_html(), _system_status_json()

        status_refresh_btn.click(
            fn=_refresh_status,
            inputs=None,
            outputs=[status_html, status_json],
            queue=False,
        )

    return demo


def main():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.queue(default_concurrency_limit=1, max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        max_threads=4,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        head=f"<script>{DARK_MODE_JS}</script>",
    )


if __name__ == "__main__":
    main()