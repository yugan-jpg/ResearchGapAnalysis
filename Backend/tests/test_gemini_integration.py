"""Opt-in smoke test. It makes one real Gemini call only when explicitly enabled."""

import os

import pytest

from app.core.config import Settings
from app.modules.claim_extraction.gemini_verifier import GeminiClaimVerifier, VerificationInput
from app.schemas.claims import ClaimCategory


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION_TEST") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="Set RUN_GEMINI_INTEGRATION_TEST=1 and GEMINI_API_KEY to make a real Gemini request.",
)
def test_gemini_verifier_live_smoke() -> None:
    settings = Settings(verifier_enabled=True)
    result = GeminiClaimVerifier(settings).verify_batch(
        [
            VerificationInput(
                claim_id="live-claim-1",
                paper_id="test-paper",
                chunk_id="test-chunk",
                page_number=1,
                claim_text="The sample included 20 participants.",
                category=ClaimCategory.METHODOLOGY,
                supporting_evidence="The sample included 20 participants.",
                source_text="The sample included 20 participants.",
                rule_completeness=True,
                rule_numerical_consistency=True,
            )
        ]
    )
    assert result["live-claim-1"].status in {"accepted", "review", "rejected"}
