"""Create a separate layout-corrected paper version. Dry-run unless --apply."""
import argparse
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymongo.errors import PyMongoError
from app.core.config import get_settings
from app.db.mongodb import close_mongo_client, get_database
from app.modules.pdf_processing.chunking import PageAwareChunker
from app.modules.pdf_processing.extraction import extract_pdf


def reingest(database, paper_id, pdf_path, *, apply=False, chunk_size=1200, overlap=200):
    original = database.papers.find_one({"paper_id": paper_id})
    if not original:
        raise ValueError("Original paper was not found.")
    with pdf_path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if not original.get("content_sha256") or digest != original["content_sha256"]:
        raise ValueError("The supplied PDF does not match the original paper's stored SHA-256.")
    extraction = extract_pdf(pdf_path)
    chunks = PageAwareChunker(chunk_size, overlap).chunk_pages(extraction.pages)
    version_id = str(uuid5(NAMESPACE_URL, f"layout-v1:{paper_id}:{digest}:{chunk_size}:{overlap}"))
    if not apply:
        return {"version_paper_id": version_id, "pages": extraction.total_pages, "chunks": len(chunks), "applied": False}
    now = datetime.now(timezone.utc)
    paper = {
        "_id": "layout-v1:" + version_id, "paper_id": version_id,
        "version_of_paper_id": paper_id, "source_content_sha256": digest,
        "layout_extraction_version": "layout-v1",
        "original_filename": original.get("original_filename", pdf_path.name),
        "file_size": pdf_path.stat().st_size, "total_pages": extraction.total_pages,
        "pages_with_text": extraction.pages_with_text, "total_chunks": len(chunks),
        "status": "processing", "created_at": now, "metadata": extraction.metadata,
    }
    database.papers.update_one({"_id": paper["_id"]}, {"$setOnInsert": paper}, upsert=True)
    for chunk in chunks:
        chunk_id = str(uuid5(NAMESPACE_URL, f"{version_id}:{chunk.chunk_index}"))
        doc = {
            "_id": "layout-v1:" + chunk_id, "chunk_id": chunk_id, "paper_id": version_id,
            "chunk_index": chunk.chunk_index, "page_number": chunk.page_number,
            "text": chunk.text, "embedding_status": "pending", "created_at": now,
        }
        database.paper_chunks.update_one({"_id": doc["_id"]}, {"$setOnInsert": doc}, upsert=True)
        saved = database.paper_chunks.find_one({"_id": doc["_id"]})
        if not saved or saved["text"] != chunk.text or saved["page_number"] != chunk.page_number:
            raise ValueError("Existing version chunk differs; no existing data was overwritten.")
    database.papers.update_one({"_id": paper["_id"], "status": "processing"}, {"$set": {"status": "processed"}})
    return {"version_paper_id": version_id, "pages": extraction.total_pages, "chunks": len(chunks), "applied": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    try:
        print(reingest(get_database(), args.paper_id, args.pdf, apply=args.apply,
                       chunk_size=settings.pdf_chunk_size, overlap=settings.pdf_chunk_overlap))
    except (PyMongoError, OSError, ValueError, RuntimeError) as error:
        if isinstance(error, PyMongoError):
            raise SystemExit("Database operation failed; the original records were not changed.")
        raise SystemExit(str(error))
    finally:
        close_mongo_client()


if __name__ == "__main__":
    main()
