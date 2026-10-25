"""Ordered map over items: a plain loop for one worker, a process pool otherwise."""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor


def map_ordered[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Results in input order. `fn` must be a picklable top-level function (Windows spawns)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunk))
