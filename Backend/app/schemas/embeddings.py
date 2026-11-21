from pydantic import BaseModel, Field


class BackfillRequest(BaseModel):
    paper_id: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=50, ge=1, le=250)
    force: bool = False


class BackfillResponse(BaseModel):
    candidate_chunks: int
    embedded_chunks: int
    remaining_chunks: int
    model_name: str
    dimensions: int


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=50)


class SemanticSearchResult(BaseModel):
    chunk_id: str
    paper_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticSearchResult]
