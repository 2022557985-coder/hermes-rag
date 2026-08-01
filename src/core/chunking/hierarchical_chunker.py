"""Hierarchical chunking: heading-based sectioning + semantic splitting."""

import re
from typing import List, Dict, Any, Optional, Iterator


class HierarchicalChunker:
    """Split documents hierarchically: by headings first, then semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        semantic_threshold: float = 0.65,
        min_chunk_size: int = 50,
        max_section_size: int = 512,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_device: str = "cpu",
    ):
        """Initialize the hierarchical chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between adjacent chunks in tokens.
            semantic_threshold: Cosine similarity threshold for semantic splits.
            min_chunk_size: Minimum tokens for a chunk.
            max_section_size: Max section tokens before triggering semantic split.
            embedding_model: SentenceTransformer model name for semantic chunking.
            embedding_device: Device for semantic chunker model ('cpu' or 'cuda').
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.min_chunk_size = min_chunk_size
        self.max_section_size = max_section_size
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self._semantic_chunker = None

    def _get_semantic_chunker(self):
        if self._semantic_chunker is None:
            from .semantic_chunker import SemanticChunker
            self._semantic_chunker = SemanticChunker(
                threshold=self.semantic_threshold,
                min_chunk_size=self.min_chunk_size,
                model_name=self.embedding_model,
                device=self.embedding_device,
            )
        return self._semantic_chunker

    def chunk(
        self,
        text: str,
        source_name: str = "",
        page_num: Optional[int] = None,
        headings: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk a document into hierarchical chunks with metadata.

        Args:
            text: Document text.
            source_name: Source file name.
            page_num: Page number (for PDFs).
            headings: Pre-extracted headings from Markdown.

        Returns:
            List of chunks, each as dict with 'text', 'metadata', 'chunk_id'.
        """
        from .metadata_builder import MetadataBuilder

        meta_builder = MetadataBuilder()
        chunks = []

        # Step 1: Split by headings
        sections = self._split_by_headings(text, headings)

        chunk_index = 0
        for section in sections:
            if section.get("heading"):
                meta_builder.update_heading(
                    section["heading"]["level"],
                    section["heading"]["title"],
                )

            section_text = section["text"]

            # Step 2: If section is too large, use semantic split
            token_count = self._estimate_tokens(section_text)
            if token_count > self.max_section_size:
                sub_chunks = self._get_semantic_chunker().split(section_text)
            else:
                sub_chunks = [section_text]

            # Step 3: Apply overlap-based chunking to each sub-chunk
            for sub_text in sub_chunks:
                if len(sub_text.strip()) < self.min_chunk_size:
                    continue

                sub_token_count = self._estimate_tokens(sub_text)
                if sub_token_count > self.chunk_size:
                    # Use sliding window with overlap
                    for overlapping_chunk in self._sliding_window_chunk(sub_text):
                        meta = meta_builder.build_metadata(
                            source_name=source_name,
                            page_num=page_num,
                            chunk_index=chunk_index,
                        )
                        chunks.append({
                            "text": overlapping_chunk,
                            "metadata": meta,
                            "chunk_id": f"{source_name}_{chunk_index}",
                        })
                        chunk_index += 1
                else:
                    meta = meta_builder.build_metadata(
                        source_name=source_name,
                        page_num=page_num,
                        chunk_index=chunk_index,
                    )
                    chunks.append({
                        "text": sub_text,
                        "metadata": meta,
                        "chunk_id": f"{source_name}_{chunk_index}",
                    })
                    chunk_index += 1

        return chunks

    def _split_by_headings(
        self,
        text: str,
        headings: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Split text into sections based on heading markers.

        If headings are pre-extracted, use them. Otherwise, detect # style headings.
        """
        sections = []

        if headings:
            # Use pre-extracted headings to find section boundaries
            positions = []
            for h in headings:
                marker = "#" * h["level"] + " " + h["title"]
                pos = text.find(marker)
                if pos >= 0:
                    positions.append((pos, h))

            if not positions:
                sections = [{"text": text}]
            else:
                # Sort by position
                positions.sort(key=lambda x: x[0])

                # Text before first heading
                if positions[0][0] > 0:
                    sections.append({"text": text[:positions[0][0]].strip()})

                # Sections between headings
                for i, (pos, h) in enumerate(positions):
                    # Find end of heading line
                    line_end = text.find("\n", pos)
                    current_marker = "#" * h["level"] + " " + h["title"]
                    content_start = line_end + 1 if line_end >= 0 else pos + len(current_marker)
                    if i + 1 < len(positions):
                        content_end = positions[i + 1][0]
                    else:
                        content_end = len(text)
                    section_text = text[content_start:content_end].strip()
                    if section_text:
                        sections.append({
                            "text": section_text,
                            "heading": h,
                        })
        else:
            # Auto-detect heading markers
            pattern = r"^(#{1,6})\s+(.+)$"
            lines = text.split("\n")
            current_section = []
            current_heading = None

            for line in lines:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section:
                        section_text = "\n".join(current_section).strip()
                        if section_text:
                            sections.append({
                                "text": section_text,
                                "heading": current_heading,
                            })
                    # Start new section
                    current_heading = {
                        "level": len(match.group(1)),
                        "title": match.group(2).strip(),
                    }
                    current_section = [line]
                else:
                    current_section.append(line)

            # Last section
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append({
                        "text": section_text,
                        "heading": current_heading,
                    })

        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{"text": text}]

        return sections

    def _sliding_window_chunk(self, text: str) -> Iterator[str]:
        """Generate overlapping chunks using sliding window based on tokens."""
        from src.utils.text_utils import tokenize_text
        tokens = tokenize_text(text)
        chunk_tokens = max(int(self.chunk_size * 0.75), 1)
        overlap_tokens = max(int(self.chunk_overlap * 0.75), 1)
        step = chunk_tokens - overlap_tokens

        if step <= 0:
            step = chunk_tokens // 2

        for i in range(0, len(tokens), step):
            chunk = " ".join(tokens[i:i + chunk_tokens])
            if chunk.strip():
                yield chunk

    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimate token count using text_utils."""
        from src.utils.text_utils import estimate_tokens
        return estimate_tokens(text)