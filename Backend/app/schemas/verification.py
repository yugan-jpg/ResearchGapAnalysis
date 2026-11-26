from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.schemas.claims import VerificationVerdict


class GroqVerificationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=100)
    verdict: VerificationVerdict
    evidence_supported: StrictBool
    numerical_consistency: StrictBool
    completeness: StrictBool
    category_correct: StrictBool
    explanation: str = Field(min_length=1, max_length=2_000)
    duplicate_of_claim_id: str | None = Field(default=None, max_length=100)
    duplicate_type: Literal["semantic", "partial_overlap", "containment", "related_distinct"] | None = None
    duplicate_explanation: str | None = Field(default=None, max_length=1000)


class GroqVerificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifications: list[GroqVerificationItem] = Field(min_length=1, max_length=20)
