import json
import time
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.modules.claim_extraction.errors import (
    ClaimResponseValidationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.schemas.claims import LLMClaimPayload


class HttpClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


class QwenClaimClient:
    """Configurable OpenAI-compatible client for a hosted Qwen model.

    The provider URL is deliberately supplied through environment settings so a
    provider can be chosen later without changing extraction code.
    """

    def __init__(self, settings: Settings, http_client: HttpClient | None = None, sleep=time.sleep) -> None:
        self.settings = settings
        self._http_client = http_client
        self._sleep = sleep

    def extract_claims(self, paper_id: str, chunk_id: str, page_number: int, chunk_text: str) -> LLMClaimPayload:
        self._validate_configuration()
        payload = {
            "model": self.settings.llm_model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific claim extraction assistant. Return only valid JSON. "
                        "Extract only statements directly supported by the supplied source chunk. "
                        "Do not infer, combine evidence, add conclusions, or paraphrase supporting evidence. "
                        "Each supporting_evidence value must be an exact continuous quotation from the chunk."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Paper ID: {paper_id}\nChunk ID: {chunk_id}\nPage: {page_number}\n\n"
                        "Source chunk:\n---\n"
                        f"{chunk_text}\n"
                        "---\n\n"
                        f"Return a JSON object only: {{\"claims\": [...]}}. Return at most {self.settings.claim_extraction_max_claims} claims. "
                        "Every item must have claim_text, category, and supporting_evidence. "
                        "category must be exactly one of: finding, methodology, limitation, uncertainty, future_research. "
                        "If the chunk has no clear supported claim, return {\"claims\": []}."
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        response_body = self._request(payload)
        return self._parse_claims(response_body)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        key = self.settings.llm_api_key.get_secret_value()
        auth_value = f"{self.settings.llm_api_auth_prefix.strip()} {key}".strip()
        headers[self.settings.llm_api_auth_header] = auth_value

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                if self._http_client is not None:
                    response = self._http_client.post(self.settings.llm_api_url, headers=headers, json=payload, timeout=self.settings.llm_timeout_seconds)
                else:
                    with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                        response = client.post(self.settings.llm_api_url, headers=headers, json=payload)
            except httpx.TimeoutException as error:
                if attempt < self.settings.llm_max_retries:
                    self._sleep(2**attempt)
                    continue
                raise LLMTimeoutError("The claim extraction provider timed out. Please try again.") from error
            except httpx.HTTPError as error:
                if attempt < self.settings.llm_max_retries:
                    self._sleep(2**attempt)
                    continue
                raise LLMProviderError("The claim extraction provider could not be reached.") from error

            if response.status_code == 429:
                if attempt < self.settings.llm_max_retries:
                    retry_after = response.headers.get("retry-after")
                    self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt)
                    continue
                raise LLMRateLimitError("The claim extraction provider is rate limiting requests. Please try again later.")
            if response.status_code >= 500:
                if attempt < self.settings.llm_max_retries:
                    self._sleep(2**attempt)
                    continue
                raise LLMProviderError("The claim extraction provider is unavailable. Please try again later.")
            if response.status_code >= 400:
                raise LLMProviderError("The claim extraction provider rejected the request. Check its endpoint, model, and credentials.")
            try:
                return response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise ClaimResponseValidationError("The claim extraction provider returned invalid JSON.") from error

        raise LLMProviderError("The claim extraction provider could not complete the request.")

    def _parse_claims(self, response_body: dict[str, Any]) -> LLMClaimPayload:
        content: Any = response_body
        if "choices" in response_body:
            try:
                content = response_body["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError) as error:
                raise ClaimResponseValidationError("The claim extraction provider response has no usable message content.") from error
        if isinstance(content, str):
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else ""
                if text.endswith("```"):
                    text = text[:-3].strip()
            try:
                content = json.loads(text)
            except json.JSONDecodeError as error:
                raise ClaimResponseValidationError("The claim extraction provider did not return valid claim JSON.") from error
        try:
            claims = LLMClaimPayload.model_validate(content)
        except ValidationError as error:
            raise ClaimResponseValidationError("The claim extraction provider returned claims in an invalid format.") from error
        if len(claims.claims) > self.settings.claim_extraction_max_claims:
            raise ClaimResponseValidationError("The claim extraction provider exceeded the configured claim limit.")
        return claims

    def _validate_configuration(self) -> None:
        if not self.settings.llm_api_url or not self.settings.llm_model_name or not self.settings.llm_api_key.get_secret_value():
            raise LLMConfigurationError(
                "Claim extraction is not configured. Set LLM_API_URL, LLM_MODEL_NAME, and LLM_API_KEY before extracting claims."
            )
