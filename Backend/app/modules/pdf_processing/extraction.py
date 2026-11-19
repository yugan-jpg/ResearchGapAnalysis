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
            text = clean_page_text(page.get_text("text", sort=True))
            if text:
                pages.append(ExtractedPage(page_number=index, text=text))

        if not pages:
            raise ScannedPdfError("No readable text was found. Scanned or image-only PDFs require OCR, which is not available yet.")
        return ExtractionResult(total_pages=document.page_count, pages=pages, metadata=_reliable_metadata(document))
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise InvalidPdfError("The PDF is corrupted or cannot be read.") from error
    finally:
        document.close()
