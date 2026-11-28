from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class Relationship(str, Enum):
    AGREEMENT = "agreement"
    DIRECT = "direct_contradiction"
    PARTIAL = "partial_contradiction"
    COMPLEMENTARY = "complementary_findings"
    CONTEXTUAL = "contextual_differences"
    INSUFFICIENT = "insufficient_evidence"


class Mismatch(str, Enum):
    CLAIM_SCOPE = "claim_scope"
    EVIDENCE_METHOD = "evidence_method"
    POPULATION_CONTEXT = "population_context"
    CROSS_FIELD = "cross_field"


class ClaimSource(BaseModel):
    claim_id: str
    paper_id: str
    chunk_id: str
    page_number: int = Field(ge=1)
    claim_text: str
    category: str
    supporting_evidence: str


class Candidate(BaseModel):
    left: ClaimSource
    right: ClaimSource
    similarity_score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    structural_signals: dict[str, list[str]]


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: Relationship
    mismatch_types: list[Mismatch] = Field(max_length=4)
    explanation: str = Field(min_length=1, max_length=3000)
    left_quote: str = Field(min_length=1, max_length=6000)
    right_quote: str = Field(min_length=1, max_length=6000)
    comparable_scope: bool = Field(strict=True)

    @model_validator(mode="after")
    def check_conclusion(self):
        if not all(x.strip() for x in (self.explanation, self.left_quote, self.right_quote)):
            raise ValueError("The comparison requires two quotations and an explanation.")
        if len(self.mismatch_types) != len(set(self.mismatch_types)):
            raise ValueError("Each mismatch type must be listed once.")
        if self.relationship in {Relationship.AGREEMENT, Relationship.DIRECT, Relationship.PARTIAL} and not self.comparable_scope:
            raise ValueError("Agreement or contradiction requires comparable scope for the finding.")
        return self


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_paper_id: Identifier
    target_paper_id: Identifier
    after_claim_id: Identifier | None = None
    limit: int | None = Field(default=None, ge=1, le=25)
    min_similarity: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def distinct_ids(self):
        if self.source_paper_id == self.target_paper_id:
            raise ValueError("Choose two different publications.")
        return self


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: Identifier
    other_paper_id: Identifier | None = None
    limit: int | None = Field(default=None, ge=1, le=25)
    retry_failed: bool = False

    @model_validator(mode="after")
    def distinct_ids(self):
        if self.paper_id == self.other_paper_id:
            raise ValueError("Choose two different publications.")
        return self


ProcessingStatus = Literal["pending", "processing", "completed", "failed", "stale"]
VerificationStatus = Literal["pending", "verified", "review", "unavailable", "stale"]


class Comparison(Candidate):
    comparison_id: str
    processing_status: ProcessingStatus
    verification_status: VerificationStatus
    relationship: Relationship | None = None
    mismatch_types: list[Mismatch] = Field(default_factory=list)
    explanation: str | None = None
    primary: Judgment | None = None
    secondary: Judgment | None = None
    primary_model: str | None = None
    verifier_model: str | None = None
    failure_reason: str | None = None
    attempts: int
    updated_at: datetime
    sources_current: bool


class DiscoveryResponse(BaseModel):
    scanned: int
    next_cursor: str | None
    queued: int
    comparisons: list[Comparison]


class DetectionResponse(BaseModel):
    attempted: int
    completed: int
    failed: int
    stale: int
    comparisons: list[Comparison]


class ComparisonList(BaseModel):
    total: int
    skip: int
    limit: int
    comparisons: list[Comparison]
