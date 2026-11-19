from io import BytesIO

import pymupdf
import pytest


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    result = document.tobytes()
    document.close()
    return result


@pytest.fixture
def pdf_factory():
    return make_pdf_bytes


@pytest.fixture
def bytes_io():
    return BytesIO
