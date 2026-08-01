"""Gradio web interface for Hermes-RAG."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr


def create_ui():
    """Create and launch the Gradio web interface."""

    def query_hermes(query: str, top_k: int, use_reranker: bool, generate_answer: bool):
        """Handle query from Gradio UI."""
        try:
            from api.routes import _get_pipeline

            pipeline = _get_pipeline()
            result = pipeline.retrieve(
                query=query,
                top_k=top_k,
                use_reranker=use_reranker,
            )

            # Format results for display
            results_text = ""
            for i, r in enumerate(result["results"], 1):
                score = r.get("score", 0)
                source = r.get("metadata", {}).get("source", "unknown")
                heading = r.get("metadata", {}).get("heading_path", "")
                page = r.get("metadata", {}).get("page", "")

                header = f"## [{i}] Score: {score:.4f}"
                if heading:
                    header += f" | {heading}"
                if page:
                    header += f" | Page {page}"
                header += f" | {source}"

                results_text += f"{header}\n\n{r.get('text', '')}\n\n---\n\n"

            # Generate answer only if requested
            answer = ""
            if generate_answer:
                try:
                    from src.config import get_config
                    from src.core.generation.llm_client import LLMClient

                    cfg = get_config()
                    provider = cfg.generation.provider
                    if provider == "openai":
                        llm = LLMClient(
                            provider="openai",
                            model=cfg.generation.openai.model,
                            base_url=cfg.generation.openai.base_url,
                            api_key=cfg.generation.openai.api_key,
                            temperature=cfg.generation.temperature,
                        )
                    else:
                        llm = LLMClient(
                            provider="ollama",
                            model=cfg.generation.ollama.model,
                            base_url=cfg.generation.ollama.base_url,
                            temperature=cfg.generation.temperature,
                        )
                    answer = llm.generate(query, result["results"])
                except Exception:
                    answer = "（LLM 服务未连接，仅展示检索结果）"

            timing = result.get("timing", {}).get("total", 0)
            timing_info = f"检索耗时: {timing:.3f}s"

            return results_text, answer, timing_info

        except Exception as e:
            return f"Error: {e}", "", ""

    with gr.Blocks(title="Hermes-RAG") as demo:
        gr.Markdown(
            """
            # Hermes-RAG
            ### 轻量级、高精度 RAG 检索优化框架
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="输入查询",
                    placeholder="请输入您的问题...",
                    lines=3,
                )
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="返回结果数 (Top-K)",
                    )
                    reranker_toggle = gr.Checkbox(
                        value=True,
                        label="启用重排序 (Reranker)",
                    )
                    gen_answer_toggle = gr.Checkbox(
                        value=False,
                        label="生成 AI 回答",
                    )
                query_btn = gr.Button("检索", variant="primary")

            with gr.Column(scale=1):
                timing_output = gr.Textbox(label="性能", interactive=False)
                answer_output = gr.Textbox(
                    label="AI 回答",
                    interactive=False,
                    lines=8,
                )

        results_output = gr.Markdown(label="检索结果")

        query_btn.click(
            fn=query_hermes,
            inputs=[query_input, top_k_slider, reranker_toggle, gen_answer_toggle],
            outputs=[results_output, answer_output, timing_output],
        )

    return demo


def main():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()