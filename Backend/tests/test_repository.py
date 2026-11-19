from pathlib import Path

import mongomock
import pytest
from pymongo.errors import PyMongoError

from app.modules.pdf_processing.chunking import TextChunk
from app.modules.pdf_processing.errors import DuplicatePaperError, StorageError
from app.modules.pdf_processing.extraction import ExtractedPage, ExtractionResult
from app.modules.pdf_processing.repository import PaperRepository
from app.modules.pdf_processing.upload import SavedUpload


def _upload() -> SavedUpload:
    return SavedUpload(Path("example.pdf"), "example.pdf", 1234, "a" * 64)


def _extraction() -> ExtractionResult:
    return ExtractionResult(1, [ExtractedPage(1, "Scientific text.")], {"title": "Example"})


def _chunks() -> list[TextChunk]:
    return [TextChunk(0, 1, "Scientific text.")]


def test_stores_chunks_and_retrieves_metadata() -> None:
    database = mongomock.MongoClient().phase2_test
    repository = PaperRepository(database)

    stored = repository.create_paper(_upload(), _extraction(), _chunks())

    assert stored["status"] == "processed"
    assert database.paper_chunks.count_documents({"paper_id": stored["paper_id"]}) == 1
    assert repository.get_paper(stored["paper_id"])["original_filename"] == "example.pdf"


def test_duplicate_hash_returns_existing_paper_id() -> None:
    repository = PaperRepository(mongomock.MongoClient().phase2_test)
    existing = repository.create_paper(_upload(), _extraction(), _chunks())

    with pytest.raises(DuplicatePaperError) as error:
        repository.create_paper(_upload(), _extraction(), _chunks())

    assert error.value.paper_id == existing["paper_id"]


def test_database_failure_is_safe() -> None:
    class FailingPapers:
        def find_one(self, *_args, **_kwargs):
            return None

        def insert_one(self, *_args, **_kwargs):
            raise PyMongoError("database unavailable")

        def delete_one(self, *_args, **_kwargs):
            return None

    class FailingChunks:
        def delete_many(self, *_args, **_kwargs):
            return None

    class FailingDatabase:
        papers = FailingPapers()
        paper_chunks = FailingChunks()

    with pytest.raises(StorageError):
        PaperRepository(FailingDatabase()).create_paper(_upload(), _extraction(), _chunks())
