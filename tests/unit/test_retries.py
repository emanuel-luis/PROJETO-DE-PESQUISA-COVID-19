import pytest

from covidxr.errors import SourceUnavailableError
from covidxr.sources._common import with_retries


def test_succeeds_after_transient_failures(_no_retry_sleep: list[float]) -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise SourceUnavailableError("TLS connect error")
        return "ok"

    assert with_retries(flaky, what="download") == "ok"
    assert len(calls) == 3
    assert _no_retry_sleep == [2.0, 4.0]


def test_gives_up_after_the_last_attempt(_no_retry_sleep: list[float]) -> None:
    def down() -> None:
        raise SourceUnavailableError("offline")

    with pytest.raises(SourceUnavailableError, match="offline"):
        with_retries(down, what="download", attempts=3)
    assert _no_retry_sleep == [2.0, 4.0]


def test_other_errors_are_not_retried(_no_retry_sleep: list[float]) -> None:
    calls: list[int] = []

    def bug() -> None:
        calls.append(1)
        raise ValueError("bad layout")

    with pytest.raises(ValueError, match="bad layout"):
        with_retries(bug, what="download")
    assert calls == [1] and _no_retry_sleep == []
