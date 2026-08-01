"""PDF parser using PyMuPDF."""

import re
from pathlib import Path
from typing import Any

from .parser_factory import BaseParser


class PDFParser(BaseParser):
    """Parse PDF documents using PyMuPDF."""

    def parse(self, source: str) -> dict[str, Any]:
        """Parse a PDF file.

        Args:
            source: Path to PDF file.

        Returns:
            dict with text, tables, metadata.
        """
        import fitz  # PyMuPDF

        doc = fitz.open(source)
        full_text = []
        tables = []
        metadata = {
            "source": Path(source).name,
            "type": "pdf",
            "total_pages": len(doc),
            "pages": [],
        }

        for page_num, page in enumerate(doc, 1):
            # Extract text
            text = page.get_text("text")
            clean_text = ""
            if text:
                # Filter header/footer (first and last lines of each page)
                lines = text.strip().split("\n")
                if len(lines) > 3:
                    # Skip potential header (first line) and footer (last line)
                    filtered_lines = lines[1:-1]
                    text = "\n".join(filtered_lines)

                clean_text = self._clean_text(text)
                full_text.append(clean_text)

            # Extract tables
            page_tables = self._extract_tables(page)
            for table_md in page_tables:
                tables.append({
                    "page": page_num,
                    "content": table_md,
                })

            metadata["pages"].append({
                "page": page_num,
                "char_count": len(clean_text) if text else 0,
            })

        doc.close()

        return {
            "text": "\n\n".join(full_text),
            "tables": tables,
            "metadata": metadata,
        }

    def _extract_tables(self, page) -> list:
        """Extract tables from a page, convert to Markdown."""
        tables = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for table in tabs.tables:
                    md_table = self._table_to_markdown(table.extract())
                    if md_table:
                        tables.append(md_table)
        except (AttributeError, ValueError, RuntimeError):
            pass
        return tables

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove watermark-like patterns
        text = re.sub(r"^\d+/\d+$", "", text, flags=re.MULTILINE)
        return text.strip()