from pathlib import Path

import pytest

from app.modules.pdf_processing.chunking import PageAwareChunker
from app.modules.pdf_processing.errors import ChunkingError, InvalidPdfError, ScannedPdfError
from app.modules.pdf_processing.extraction import ExtractedPage, clean_page_text, extract_pdf


def test_extracts_multiple_pages_and_preserves_page_numbers(tmp_path: Path, pdf_factory) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(pdf_factory(["First page methods and values.", "Second page findings."]))

    result = extract_pdf(pdf_path)

    assert result.total_pages == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "methods" in result.pages[0].text
    assert result.pages_with_text == 2


def test_cleaning_preserves_scientific_unicode() -> None:
    assert clean_page_text("α and μ\n\nvalues") == "α and μ\n\nvalues"


def test_rejects_invalid_pdf(tmp_path: Path) -> None:
    invalid = tmp_path / "not-a-pdf.pdf"
    invalid.write_bytes(b"not a pdf")

    with pytest.raises(InvalidPdfError):
        extract_pdf(invalid)


def test_rejects_pdf_without_extractable_text(tmp_path: Path, pdf_factory) -> None:
    scanned = tmp_path / "image-only.pdf"
    scanned.write_bytes(pdf_factory([""]))

    with pytest.raises(ScannedPdfError):
        extract_pdf(scanned)


def test_chunker_keeps_pages_order_and_overlap() -> None:
    text = " ".join(f"Sentence {number}." for number in range(300))
    chunks = PageAwareChunker(chunk_size=400, overlap=100).chunk_pages([
        ExtractedPage(page_number=3, text=text),
        ExtractedPage(page_number=4, text="Final page text."),
    ])

    assert len(chunks) > 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks[:-1])
    assert chunks[-1].page_number == 4
    assert chunks[0].text[-50:] in chunks[1].text


@pytest.mark.parametrize("size,overlap", [(199, 0), (300, -1), (300, 300)])
def test_chunker_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ChunkingError):
        PageAwareChunker(size, overlap)
