class ClaimExtractionError(Exception):
    """Base error for scientific claim extraction."""


class LLMConfigurationError(ClaimExtractionError):
    """The hosted LLM provider settings are incomplete."""


class LLMTimeoutError(ClaimExtractionError):
    """The hosted LLM provider did not respond in time."""


class LLMRateLimitError(ClaimExtractionError):
    """The hosted LLM provider rejected the request due to rate limiting."""


class LLMProviderError(ClaimExtractionError):
    """The hosted LLM provider returned an unavailable or invalid response."""


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected authentication or model access."""


class ClaimResponseValidationError(ClaimExtractionError):
    """The LLM response could not be parsed into the required claim schema."""


class EvidenceVerificationError(ClaimExtractionError):
    """A claim's purported supporting passage is absent from its source chunk."""


class ClaimStorageError(ClaimExtractionError):
    """MongoDB could not safely persist claim extraction results."""


class PaperNotFoundError(ClaimExtractionError):
    """The selected paper does not exist in MongoDB."""
