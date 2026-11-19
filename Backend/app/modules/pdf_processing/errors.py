class PdfProcessingError(Exception):
    """Base error for safe, user-facing PDF ingestion failures."""


class UploadValidationError(PdfProcessingError):
    """The uploaded file is unsafe, unsupported, or invalid."""


class FileTooLargeError(UploadValidationError):
    """The upload exceeds the configured size limit."""


class InvalidPdfError(PdfProcessingError):
    """The file is not a readable PDF document."""


class EmptyPdfError(PdfProcessingError):
    """The PDF contains no pages."""


class PasswordProtectedPdfError(PdfProcessingError):
    """The PDF needs a password and cannot be ingested safely."""


class ScannedPdfError(PdfProcessingError):
    """The PDF has no extractable text and requires OCR, which is deferred."""


class ChunkingError(PdfProcessingError):
    """Chunking configuration or input text is invalid."""


class DuplicatePaperError(PdfProcessingError):
    def __init__(self, paper_id: str):
        super().__init__("This PDF has already been ingested.")
        self.paper_id = paper_id


class StorageError(PdfProcessingError):
    """MongoDB could not safely store a processed paper."""
