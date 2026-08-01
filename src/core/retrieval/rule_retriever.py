"""Rule-based retriever for chapter/section indicators."""

import re
from typing import List, Dict, Any, Optional


class RuleRetriever:
    """Regex-based rule engine for chapter/section pre-filtering."""

    # Patterns for chapter/section indicators
    CHAPTER_PATTERNS = [
        r"第[一二三四五六七八九十\d]+章",
        r"第[一二三四五六七八九十\d]+节",
        r"附录\s*[A-Za-z]",
        r"Chapter\s+\d+",
        r"Section\s+\d+",
        r"Part\s+\d+",
    ]

    def detect_chapter_hint(self, query: str) -> Optional[str]:
        """Detect chapter/section indicators in query.

        Args:
            query: Query text.

        Returns:
            Matched chapter hint or None.
        """
        for pattern in self.CHAPTER_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group()
        return None

    def build_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """Build a metadata filter based on query content.

        Args:
            query: Query text.

        Returns:
            ChromaDB-compatible filter dict or None.
        """
        chapter_hint = self.detect_chapter_hint(query)
        if chapter_hint:
            return {"heading_path": {"$contains": chapter_hint}}
        return None