from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.modules.pdf_processing.chunking import TextChunk
from app.modules.pdf_processing.errors import DuplicatePaperError, StorageError
from app.modules.pdf_processing.extraction import ExtractionResult
from app.modules.pdf_processing.upload import SavedUpload


class PaperRepository:
    """MongoDB persistence with compensating cleanup for partial inserts."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_paper(self, upload: SavedUpload, extraction: ExtractionResult, chunks: list[TextChunk]) -> dict[str, Any]:
        existing = self.database.papers.find_one({"content_sha256": upload.sha256_hash}, {"paper_id": 1})
        if existing:
            raise DuplicatePaperError(existing["paper_id"])

        paper_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        paper_document = {
            "paper_id": paper_id,
            "original_filename": upload.filename,
            "file_size": upload.file_size,
            "content_sha256": upload.sha256_hash,
            "total_pages": extraction.total_pages,
            "pages_with_text": extraction.pages_with_text,
            "total_chunks": len(chunks),
            "status": "processing",
            "created_at": created_at,
            "metadata": extraction.metadata,
        }
        chunk_documents = [
            {
                "chunk_id": str(uuid4()),
                "paper_id": paper_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "embedding_status": "pending",
                "created_at": created_at,
            }
            for chunk in chunks
        ]
        paper_inserted = False
        try:
            self.database.papers.insert_one(paper_document)
            paper_inserted = True
            self.database.paper_chunks.insert_many(chunk_documents, ordered=True)
            self.database.papers.update_one({"paper_id": paper_id}, {"$set": {"status": "processed"}})
        except DuplicateKeyError as error:
            self._cleanup_partial(paper_id, paper_inserted)
            existing = self.database.papers.find_one({"content_sha256": upload.sha256_hash}, {"paper_id": 1})
            if existing:
                raise DuplicatePaperError(existing["paper_id"]) from error
            raise StorageError("The paper could not be stored safely. Please retry.") from error
        except PyMongoError as error:
            self._cleanup_partial(paper_id, paper_inserted)
            raise StorageError("The database could not store the processed PDF. Please retry.") from error
        return {**paper_document, "status": "processed"}

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self.database.papers.find_one({"paper_id": paper_id}, {"_id": 0})

    def list_papers(self, skip: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        projection = {"_id": 0, "paper_id": 1, "original_filename": 1, "file_size": 1, "content_sha256": 1, "total_pages": 1, "pages_with_text": 1, "total_chunks": 1, "status": 1, "created_at": 1, "metadata": 1}
        papers = list(self.database.papers.find({}, projection).sort("created_at", -1).skip(skip).limit(limit))
        return papers, self.database.papers.count_documents({})

    def _cleanup_partial(self, paper_id: str, paper_inserted: bool) -> None:
        try:
            self.database.paper_chunks.delete_many({"paper_id": paper_id})
            if paper_inserted:
                self.database.papers.delete_one({"paper_id": paper_id})
        except PyMongoError:
            # The original storage failure is the safe client-facing error. A
            # non-processed record is preferable to a misleading completed one.
            pass
