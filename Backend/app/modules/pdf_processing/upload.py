from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile

from fastapi import UploadFile

from app.modules.pdf_processing.errors import FileTooLargeError, UploadValidationError

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}
_STREAM_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    filename: str
    file_size: int
    sha256_hash: str


def validate_upload_metadata(upload: UploadFile | None) -> str:
    if upload is None:
        raise UploadValidationError("A PDF file is required.")
    filename = (upload.filename or "").strip()
    if not filename:
        raise UploadValidationError("The uploaded file must have a filename.")
    if any(separator in filename for separator in ("/", "\\")) or filename in {".", ".."}:
        raise UploadValidationError("The uploaded filename is invalid.")
    if len(filename) > 255 or not filename.lower().endswith(".pdf"):
        raise UploadValidationError("Only PDF files are supported.")
    if upload.content_type and upload.content_type.lower() not in _ALLOWED_CONTENT_TYPES:
        raise UploadValidationError("The uploaded file must use a PDF content type.")
    return filename


async def stream_upload_to_temporary_file(upload: UploadFile, filename: str, max_bytes: int) -> SavedUpload:
    """Stream an upload in bounded pieces and validate the PDF signature."""
    temporary = tempfile.NamedTemporaryFile(prefix="literature-paper-", suffix=".pdf", delete=False)
    file_size = 0
    digest = sha256()
    first_chunk = b""
    try:
        with temporary:
            while True:
                chunk = await upload.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                if not first_chunk:
                    first_chunk = chunk
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise FileTooLargeError("PDF uploads are limited to 20 MB.")
                digest.update(chunk)
                temporary.write(chunk)
        if file_size == 0:
            raise UploadValidationError("The uploaded PDF is empty.")
        if not first_chunk.startswith(b"%PDF-"):
            raise UploadValidationError("The uploaded file does not have a valid PDF signature.")
        return SavedUpload(path=Path(temporary.name), filename=filename, file_size=file_size, sha256_hash=digest.hexdigest())
    except Exception:
        Path(temporary.name).unlink(missing_ok=True)
        raise
