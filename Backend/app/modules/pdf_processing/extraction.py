from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import pymupdf

from app.modules.pdf_processing.errors import (
    EmptyPdfError,
    InvalidPdfError,
    PasswordProtectedPdfError,
    ScannedPdfError,
)


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractionResult:
    total_pages: int
    pages: list[ExtractedPage]
    metadata: dict[str, str]

    @property
    def pages_with_text(self) -> int:
        return len(self.pages)


def clean_page_text(raw_text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries and Unicode."""
    normalized = unicodedata.normalize("NFC", raw_text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    current_lines: list[str] = []
    for line in normalized.split("\n"):
        line = re.sub(r"[\t \f\v]+", " ", line).strip()
        if line:
            current_lines.append(line)
        elif current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []
    if current_lines:
        paragraphs.append(" ".join(current_lines))
    return "\n\n".join(paragraphs)


_SUPER_DIGITS = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")


def _layout_text(page: pymupdf.Page) -> str:
    """Read column blocks top-to-bottom, keeping full-width headings in place.

    PDF geometry is heuristic: blocks crossing the midpoint separate horizontal
    sections; within each section left and right column blocks are read in
    order. Superscript digits stay distinguishable from the adjacent text.
    """
    blocks = []
    for block in page.get_text("dict", sort=False)["blocks"]:
        if block.get("type") != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                value = span.get("text", "")
                if span.get("flags", 0) & 1:
                    value = value.translate(_SUPER_DIGITS)
                parts.append(value)
            if parts:
                lines.append("".join(parts))
        text = "\n".join(lines).strip()
        if text:
            blocks.append((tuple(block["bbox"]), text))

    mid = (page.rect.x0 + page.rect.x1) / 2
    gap = page.rect.width * 0.015
    spanning = [b for b in blocks if b[0][0] < mid - gap and b[0][2] > mid + gap and b[0][2] - b[0][0] > page.rect.width * 0.55]
    columns = [b for b in blocks if b not in spanning]
    spanning.sort(key=lambda b: (b[0][1], b[0][0]))
    ordered = []
    previous_y = float("-inf")
    for boundary in spanning + [None]:
        next_y = boundary[0][1] if boundary else float("inf")
        band = [b for b in columns if previous_y <= (b[0][1] + b[0][3]) / 2 < next_y]
        # Group horizontally overlapping blocks, allowing more than two
        # columns; short indented paragraphs remain in their parent column.
        groups = []
        for block in sorted(band, key=lambda b: b[0][0]):
            if not groups or block[0][0] > groups[-1][0] + gap:
                groups.append([block[0][2], [block]])
            else:
                groups[-1][0] = max(groups[-1][0], block[0][2])
                groups[-1][1].append(block)
        for _, group in groups:
            ordered.extend(sorted(group, key=lambda b: (b[0][1], b[0][0])))
        if boundary:
            ordered.append(boundary)
            previous_y = next_y
    return "\n\n".join(text for _, text in ordered)


def _reliable_metadata(document: pymupdf.Document) -> dict[str, str]:
    metadata = document.metadata or {}
    result: dict[str, str] = {}
    for source_key, target_key in (("title", "title"), ("author", "author")):
        value = (metadata.get(source_key) or "").strip()
        if value and value.lower() not in {"untitled", "unknown"}:
            result[target_key] = value[:500]
    return result


def extract_pdf(path: Path) -> ExtractionResult:
    """Extract readable text, preserving 1-based source PDF page numbers."""
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError, OSError) as error:
        raise InvalidPdfError("The uploaded file is not a valid, readable PDF.") from error

    try:
        if document.needs_pass:
            raise PasswordProtectedPdfError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise EmptyPdfError("The PDF contains no pages.")

        pages: list[ExtractedPage] = []
        for index, page in enumerate(document, start=1):
            text = clean_page_text(_layout_text(page))
            if text:
                pages.append(ExtractedPage(page_number=index, text=text))

        if not pages:
            raise ScannedPdfError("No readable text was found. Scanned or image-only PDFs require OCR, which is not available yet.")
        return ExtractionResult(total_pages=document.page_count, pages=pages, metadata=_reliable_metadata(document))
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise InvalidPdfError("The PDF is corrupted or cannot be read.") from error
    finally:
        document.close()
