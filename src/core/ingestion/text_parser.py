"""Text and Markdown parser."""

import re
from pathlib import Path
from typing import Any

from .parser_factory import BaseParser


class TextParser(BaseParser):
    """Parse plain text and Markdown files."""

    def parse(self, source: str) -> dict[str, Any]:
        """Parse a text or Markdown file.

        Args:
            source: Path to text file.

        Returns:
            dict with text, tables, metadata.
        """
        import charset_normalizer

        # Detect encoding
        with open(source, "rb") as f:
            raw = f.read(10000)
        result = charset_normalizer.detect(raw)
        encoding = result["encoding"] or "utf-8"

        with open(source, encoding=encoding, errors="replace") as f:
            content = f.read()

        ext = Path(source).suffix.lower()
        is_md = ext in (".md", ".markdown")

        metadata = {
            "source": Path(source).name,
            "type": "markdown" if is_md else "text",
            "encoding": encoding,
            "char_count": len(content),
        }

        if is_md:
            # Extract Markdown table structure
            metadata["headings"] = self._extract_headings(content)

        return {
            "text": content,
            "tables": [],
            "metadata": metadata,
        }

    def _extract_headings(self, content: str) -> list:
        """Extract heading structure from Markdown."""
        headings = []
        for line in content.split("\n"):
            match = re.match(r"^(#{1,6})\s+(.+)", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append({"level": level, "title": title})
        return headings