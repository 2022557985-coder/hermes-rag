"""Tests for document ingestion module."""

import pytest
from src.core.ingestion.parser_factory import ParserFactory, BaseParser


class TestParserFactory:
    """Tests for ParserFactory."""

    def test_get_parser_pdf(self):
        parser = ParserFactory.get_parser("test.pdf")
        from src.core.ingestion.pdf_parser import PDFParser
        assert isinstance(parser, PDFParser)

    def test_get_parser_docx(self):
        parser = ParserFactory.get_parser("test.docx")
        from src.core.ingestion.docx_parser import DocxParser
        assert isinstance(parser, DocxParser)

    def test_get_parser_pptx(self):
        parser = ParserFactory.get_parser("test.pptx")
        from src.core.ingestion.pptx_parser import PptxParser
        assert isinstance(parser, PptxParser)

    def test_get_parser_txt(self):
        parser = ParserFactory.get_parser("test.txt")
        from src.core.ingestion.text_parser import TextParser
        assert isinstance(parser, TextParser)

    def test_get_parser_md(self):
        parser = ParserFactory.get_parser("test.md")
        from src.core.ingestion.text_parser import TextParser
        assert isinstance(parser, TextParser)

    def test_get_parser_web(self):
        parser = ParserFactory.get_parser("https://example.com")
        from src.core.ingestion.web_parser import WebParser
        assert isinstance(parser, WebParser)

    def test_get_parser_unknown_extension(self):
        parser = ParserFactory.get_parser("test.xyz")
        from src.core.ingestion.text_parser import TextParser
        assert isinstance(parser, TextParser)


class TestTextParser:
    """Tests for TextParser."""

    def test_parse_markdown_extracts_headings(self, sample_text, sample_docs_dir):
        import tempfile
        from src.core.ingestion.text_parser import TextParser

        md_path = sample_docs_dir + "/test_sample.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(sample_text)

        parser = TextParser()
        result = parser.parse(md_path)

        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["type"] == "markdown"
        assert "headings" in result["metadata"]
        assert len(result["metadata"]["headings"]) > 0

    def test_parse_text_handles_encoding(self, sample_docs_dir):
        from src.core.ingestion.text_parser import TextParser

        txt_path = sample_docs_dir + "/test_sample.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Hello World\nTest content")

        parser = TextParser()
        result = parser.parse(txt_path)

        assert "Hello World" in result["text"]
        assert result["metadata"]["type"] == "text"


class TestPDFParser:
    """Tests for PDFParser."""

    def test_parser_has_parse_method(self):
        from src.core.ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        assert hasattr(parser, "parse")
        assert callable(parser.parse)


class TestDocxParser:
    """Tests for DocxParser."""

    def test_parser_has_parse_method(self):
        from src.core.ingestion.docx_parser import DocxParser
        parser = DocxParser()
        assert hasattr(parser, "parse")
        assert callable(parser.parse)