from dataclasses import dataclass
from typing import Literal

ErrorStage = Literal["identity", "discovery", "download", "extract", "ocr", "translate"]


@dataclass
class DataSourceError:
    """Returned (never raised) by data-source clients on failure.

    error_code is one of: "not_found" (no data / unknown ticker),
    "network" (HTTP/transport failure — retryable), "rate_limit"
    (provider throttling — retryable), "parse" (response fetched but could
    not be interpreted), or "stale_data". Filing clients also identify the
    failed pipeline stage and source when known.
    """

    error_code: str
    message: str
    stage: ErrorStage | None = None
    source: str | None = None
