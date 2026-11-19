from dataclasses import dataclass
import re

from app.modules.pdf_processing.errors import ChunkingError
from app.modules.pdf_processing.extraction import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int
    text: str


class PageAwareChunker:
    """Split each source page independently so chunks never cross page bounds."""

    def __init__(self, chunk_size: int = 1_200, overlap: int = 200) -> None:
        if not 200 <= chunk_size <= 5_000:
            raise ChunkingError("Chunk size must be between 200 and 5000 characters.")
        if overlap < 0 or overlap >= chunk_size:
            raise ChunkingError("Chunk overlap must be zero or greater and smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        next_index = 0
        for page in pages:
            for text in self._split_page(page.text):
                chunks.append(TextChunk(chunk_index=next_index, page_number=page.page_number, text=text))
                next_index += 1
        if not chunks:
            raise ChunkingError("No text was available for chunking.")
        return chunks

    def _split_page(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        result: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                end = self._preferred_break(text, start, end)
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            if end >= length:
                break
            start = max(end - self.overlap, start + 1)
            while start < length and text[start].isspace():
                start += 1
        return result

    def _preferred_break(self, text: str, start: int, end: int) -> int:
        """Prefer paragraph then sentence boundaries in the latter half of a window."""
        lower_bound = start + (self.chunk_size // 2)
        window = text[lower_bound:end]
        paragraph = window.rfind("\n\n")
        if paragraph >= 0:
            return lower_bound + paragraph + 2
        sentence_matches = list(re.finditer(r"[.!?][\"')\]]?\s+", window))
        if sentence_matches:
            return lower_bound + sentence_matches[-1].end()
        space = window.rfind(" ")
        return lower_bound + space + 1 if space >= 0 else end
