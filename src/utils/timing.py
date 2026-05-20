from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed():
    start = time.perf_counter()
    state = {"seconds": 0.0}
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start

