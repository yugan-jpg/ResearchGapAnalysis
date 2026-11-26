"""Opt-in two-request Groq smoke check; disabled in normal test runs."""
import os
import pytest
import httpx

from app.core.config import Settings
from app.modules.claim_extraction.llm_client import QwenClaimClient
from app.modules.claim_extraction.groq_verifier import GroqClaimVerifier, VerificationInput


@pytest.mark.skipif(os.getenv("RUN_GROQ_QUALITY_SMOKE") != "1", reason="Live Groq smoke test is opt-in.")
def test_live_extraction_and_verification():
    settings = Settings()
    assert settings.llm_api_key.get_secret_value() and settings.verifier_model_name
    assert settings.llm_api_url.rstrip("/") == "https://api.groq.com/openai/v1/chat/completions"
    # Check model access before sending text; this does not establish billing status.
    response = httpx.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    available = {item["id"] for item in response.json()["data"]}
    assert {settings.llm_model_name, settings.verifier_model_name} <= available
    source = "In 2024, the study enrolled 20 participants in the intervention group."
    extracted = QwenClaimClient(settings).extract_claims("test-paper", "test-chunk", 1, source)
    assert extracted.claims
    claim = extracted.claims[0]
    assert claim.supporting_evidence in source
    item = VerificationInput(
        claim_id="test-claim", paper_id="test-paper", chunk_id="test-chunk", page_number=1,
        claim_text=claim.claim_text, category=claim.category, supporting_evidence=claim.supporting_evidence,
        source_text=source, rule_completeness=True, rule_numerical_consistency=True,
    )
    assert GroqClaimVerifier(settings).verify_batch([item])["test-claim"].status in {"accepted", "review", "rejected"}
