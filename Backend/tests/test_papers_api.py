import asyncio
from pathlib import Path

import mongomock
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from app.api.routes import papers as papers_route
from app.modules.pdf_processing.chunking import TextChunk
from app.modules.pdf_processing.extraction import ExtractedPage, ExtractionResult
from app.modules.pdf_processing.repository import PaperRepository
from app.modules.pdf_processing.upload import SavedUpload
from app.modules.pdf_processing.upload import stream_upload_to_temporary_file


def _upload_file(bytes_io, name: str, content: bytes, content_type: str = "application/pdf") -> UploadFile:
    return UploadFile(file=bytes_io(content), filename=name, headers=Headers({"content-type": content_type}))


def test_upload_duplicate_and_metadata_lookup(monkeypatch, pdf_factory, bytes_io) -> None:
    database = mongomock.MongoClient().phase2_api_test
    monkeypatch.setattr(papers_route, "get_database", lambda: database)
    content = pdf_factory(["Research abstract and methods."])

    result = asyncio.run(papers_route.upload_paper(_upload_file(bytes_io, "study.pdf", content)))
    assert result.status == "processed"

    metadata = papers_route.get_paper(result.paper_id)
    assert metadata.original_filename == "study.pdf"
    assert metadata.filename == "study.pdf"

    with pytest.raises(HTTPException) as error:
        asyncio.run(papers_route.upload_paper(_upload_file(bytes_io, "study.pdf", content)))
    assert error.value.status_code == 409


def test_rejects_invalid_signature(bytes_io) -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(papers_route.upload_paper(_upload_file(bytes_io, "study.pdf", b"not a PDF")))
    assert error.value.status_code == 400


def test_lists_papers_with_frontend_filename(monkeypatch) -> None:
    database = mongomock.MongoClient().phase2_list_test
    repository = PaperRepository(database)
    stored = repository.create_paper(
        SavedUpload(Path("paper.pdf"), "paper.pdf", 55, "b" * 64),
        ExtractionResult(1, [ExtractedPage(1, "Text")], {}),
        [TextChunk(0, 1, "Text")],
    )
    monkeypatch.setattr(papers_route, "get_database", lambda: database)

    result = papers_route.list_papers(skip=0, limit=20)

    assert result.total == 1
    assert result.papers[0].paper_id == stored["paper_id"]
    assert result.papers[0].filename == "paper.pdf"


def test_rejects_oversized_stream(bytes_io) -> None:
    file = _upload_file(bytes_io, "study.pdf", b"%PDF-" + (b"x" * 1024))
    with pytest.raises(Exception) as error:
        asyncio.run(stream_upload_to_temporary_file(file, "study.pdf", max_bytes=100))
    assert "20 MB" in str(error.value)
