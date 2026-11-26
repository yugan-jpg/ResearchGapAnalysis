from app.modules.claim_extraction.errors import ClaimExtractionError


class GroqVerificationError(ClaimExtractionError):
    """Base error for the secondary Groq verification provider."""


class GroqConfigurationError(GroqVerificationError):
    """Groq verification was enabled without complete configuration."""


class GroqTimeoutError(GroqVerificationError):
    """Groq did not respond before the configured timeout."""


class GroqRateLimitError(GroqVerificationError):
    """Groq declined a request because of a rate limit."""


class GroqProviderError(GroqVerificationError):
    """Groq was unavailable or returned an unusable provider response."""


class GroqAuthenticationError(GroqVerificationError):
    """Groq rejected the configured credential or model access."""


class GroqResponseValidationError(GroqVerificationError):
    """Groq did not return the required structured verification JSON."""
