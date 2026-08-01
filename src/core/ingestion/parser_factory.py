"""Parser factory for document ingestion."""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class BaseParser:
    """Base class for all document parsers."""
    
    def parse(self, source: str) -> Dict[str, Any]:
        """Parse a document and return structured content.

        Args:
            source: File path or URL.

        Returns:
            dict with keys: text, tables, metadata.
        """
        raise NotImplementedError

    @staticmethod
    def _table_to_markdown(data) -> str:
        """Convert table data (list of rows) to Markdown format.

        Works with both list-of-lists and python-pptx table objects.
        """
        if hasattr(data, "rows"):
            # python-pptx Table object
            rows = []
            for row in data.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                rows.append(row_data)
        else:
            rows = data

        if not rows:
            return ""

        lines = []
        # Header
        lines.append("| " + " | ".join(str(c) for c in rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        # Data rows
        for row in rows[1:]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

        return "\n".join(lines)


class ParserFactory:
    """Factory that routes to the correct parser based on file extension."""

    _parsers: Dict[str, type] = {}

    @classmethod
    def register(cls, extensions: list, parser_class: type):
        """Register a parser for given file extensions."""
        for ext in extensions:
            cls._parsers[ext.lower()] = parser_class

    @classmethod
    def get_parser(cls, source: str) -> BaseParser:
        """Get the appropriate parser for a source.

        Args:
            source: File path or URL.

        Returns:
            BaseParser instance.
        """
        # Check if it's a URL
        if source.startswith(("http://", "https://")):
            from .web_parser import WebParser
            return WebParser()

        ext = Path(source).suffix.lower()
        if ext in cls._parsers:
            return cls._parsers[ext]()

        # Fallback to text parser
        from .text_parser import TextParser
        return TextParser()


# Register built-in parsers
def _register_parsers():
    from .pdf_parser import PDFParser
    from .docx_parser import DocxParser
    from .pptx_parser import PptxParser
    from .text_parser import TextParser

    ParserFactory.register([".pdf"], PDFParser)
    ParserFactory.register([".docx"], DocxParser)
    ParserFactory.register([".pptx", ".ppt"], PptxParser)
    ParserFactory.register([".txt", ".md", ".markdown", ".rst"], TextParser)


_register_parsers()