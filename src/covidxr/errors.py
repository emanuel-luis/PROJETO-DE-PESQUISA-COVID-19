"""Exception hierarchy. The CLI turns any CovidxrError into one log line and exit code 1."""


class CovidxrError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(CovidxrError):
    """Invalid or missing params."""


class ContractError(CovidxrError):
    """A table violates its contract."""


class SourceUnavailableError(CovidxrError):
    """A data source cannot be fetched (network, credentials, layout)."""


class ImageDecodeError(CovidxrError):
    """Image bytes cannot be decoded."""
