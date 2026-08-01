"""Metadata builder for chunks."""

from typing import Dict, Any, Optional


class MetadataBuilder:
    """Build rich metadata for each chunk."""

    def __init__(self):
        self._heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]

    def update_heading(self, level: int, title: str) -> None:
        """Update the heading stack when a new heading is encountered."""
        # Pop headings at same or higher level
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, title))

    def build_metadata(
        self,
        source_name: str = "",
        page_num: Optional[int] = None,
        chunk_index: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build metadata dict for a chunk.

        Args:
            source_name: Source file name.
            page_num: Page number (for PDFs).
            chunk_index: Chunk index within the document.
            extra: Additional metadata.

        Returns:
            Metadata dict.
        """
        meta = {
            "source": source_name,
            "chunk_index": chunk_index,
        }

        if page_num is not None:
            meta["page"] = page_num

        # Build heading path
        if self._heading_stack:
            heading_path = " > ".join(title for _, title in self._heading_stack)
            meta["heading_path"] = heading_path
            meta["heading_stack"] = [{"level": lv, "title": t} for lv, t in self._heading_stack]
            # Store all heading levels dynamically
            for i, (_, title) in enumerate(self._heading_stack):
                meta[f"heading_level_{i + 1}"] = title

        if extra:
            meta.update(extra)

        return meta

    def reset(self) -> None:
        """Reset the heading stack."""
        self._heading_stack.clear()