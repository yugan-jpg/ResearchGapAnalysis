from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ClaimCategory(str, Enum):
    FINDING = "finding"
    METHODOLOGY = "methodology"
    LIMITATION = "limitation"
    UNCERTAINTY = "uncertainty"
    ASSUMPTION = "assumption"
    FUTURE_RESEARCH = "future_research"


class VerificationVerdict(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class ExtractedClaim(BaseModel):
    claim_text: str = Field(min_length=3, max_length=4_000)
    category: ClaimCategory
    supporting_evidence: str = Field(min_length=3, max_length=6_000)


class LLMClaimPayload(BaseModel):
    claims: list[ExtractedClaim] = Field(max_length=20)


class ClaimExtractionRequest(BaseModel):
    max_chunks: int = Field(default=25, ge=1, le=100)
    force: bool = False


class ClaimVerificationRequest(BaseModel):
    limit: int = Field(default=25, ge=1, le=100)
    force: bool = False


class ClaimResponse(BaseModel):
    claim_id: str
    paper_id: str
    chunk_id: str
    page_number: int
    claim_text: str
    category: ClaimCategory
    supporting_evidence: str
    created_at: datetime
    verification_status: str = "pending"
    reverification_required: bool = False
    verification_verdict: VerificationVerdict | None = None
    factual_verdict: VerificationVerdict | None = None
    evidence_supported: bool | None = None
    numerical_consistency: bool | None = None
    completeness: bool | None = None
    category_correct: bool | None = None
    verification_explanation: str | None = None
    verifier_model: str | None = None
    verification_attempted_at: datetime | None = None
    verification_failure_reason: str | None = None
    verified_at: datetime | None = None
    duplicate_of_claim_id: str | None = None
    duplicate_type: str | None = None
    duplicate_similarity: float | None = None
    duplicate_explanation: str | None = None
    duplicate_candidate_id: str | None = None
    duplicate_assessment: str | None = None
    source_references: list[dict] = Field(default_factory=list)


class ClaimExtractionResponse(BaseModel):
    paper_id: str
    processed_chunks: int
    accepted_claims: int
    duplicate_claims: int
    rejected_claims: int
    remaining_chunks: int
    claims: list[ClaimResponse]
    generated_claims: int = 0
    stored_claims: int = 0
    extraction_rejections: int = 0
    exact_duplicates_skipped: int = 0
    semantic_duplicates_flagged: int = 0
    partial_overlaps_flagged: int = 0
    verification_accepted: int = 0
    verification_review: int = 0
    verification_rejected: int = 0
    verification_unavailable: int = 0


class ClaimListResponse(BaseModel):
    paper_id: str
    claims: list[ClaimResponse]
    total: int
    skip: int
    limit: int


class ClaimVerificationResponse(BaseModel):
    paper_id: str
    processed_claims: int
    accepted_claims: int
    review_claims: int
    rejected_claims: int
    unavailable_claims: int
    claims: list[ClaimResponse]
