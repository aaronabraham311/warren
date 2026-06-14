from dataclasses import dataclass


@dataclass
class DataSourceError:
    """Returned (never raised) by data-source clients on failure.

    error_code is one of: "not_found" (no data / unknown ticker),
    "network" (any HTTP/transport failure — retryable), "parse"
    (response fetched but could not be interpreted).
    """

    error_code: str
    message: str
