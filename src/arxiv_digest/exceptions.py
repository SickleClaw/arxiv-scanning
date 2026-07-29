"""Domain-specific exceptions with actionable user-facing messages."""


class ArxivDigestError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(ArxivDigestError):
    """Raised when application configuration cannot be loaded or validated."""


class ArxivClientError(ArxivDigestError):
    """Raised when the arXiv API cannot provide valid candidate data."""


class ArxivHTTPError(ArxivClientError):
    """Raised after an arXiv HTTP request fails permanently."""


class ArxivFeedError(ArxivClientError):
    """Raised when an arXiv Atom response is malformed or incomplete."""


class HistoryError(ArxivDigestError):
    """Raised when recommendation history is invalid or cannot be persisted safely."""


class SummaryProviderError(ArxivDigestError):
    """Raised when a summary provider cannot return a valid grounded summary."""


class ReportingError(ArxivDigestError):
    """Raised when a weekly report cannot be rendered or written safely."""


class DeliveryError(ArxivDigestError):
    """Raised when an optional report delivery attempt fails safely."""
