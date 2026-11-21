from app.modules.claim_extraction.errors import ClaimExtractionError


class GeminiVerificationError(ClaimExtractionError):
    """Base error for the secondary Gemini verification provider."""


class GeminiConfigurationError(GeminiVerificationError):
    """Gemini verification was enabled without complete configuration."""


class GeminiTimeoutError(GeminiVerificationError):
    """Gemini did not respond before the configured timeout."""


class GeminiRateLimitError(GeminiVerificationError):
    """Gemini declined a request because of a rate limit."""


class GeminiProviderError(GeminiVerificationError):
    """Gemini was unavailable or returned an unusable provider response."""


class GeminiResponseValidationError(GeminiVerificationError):
    """Gemini did not return the required structured verification JSON."""
