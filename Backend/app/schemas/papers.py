from datetime import datetime

from pydantic import BaseModel, Field


class PaperUploadResponse(BaseModel):
    message: str = "PDF processed successfully"
    paper_id: str
    filename: str
    total_pages: int
    pages_with_text: int
    total_chunks: int
    status: str


class PaperMetadataResponse(BaseModel):
    paper_id: str
    filename: str
    original_filename: str
    file_size: int
    content_sha256: str | None = None
    total_pages: int
    pages_with_text: int
    total_chunks: int
    status: str
    created_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)


class PaperListResponse(BaseModel):
    papers: list[PaperMetadataResponse]
    total: int
    skip: int
    limit: int
