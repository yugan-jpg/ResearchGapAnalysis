from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.schemas.claims import VerificationVerdict


class GeminiVerificationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=100)
    verdict: VerificationVerdict
    evidence_supported: StrictBool
    numerical_consistency: StrictBool
    completeness: StrictBool
    category_correct: StrictBool
    explanation: str = Field(min_length=1, max_length=2_000)
    duplicate_of_claim_id: str | None = Field(default=None, max_length=100)


class GeminiVerificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifications: list[GeminiVerificationItem] = Field(min_length=1, max_length=20)
