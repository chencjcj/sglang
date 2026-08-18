"""Wall-clock stage timing for the multimodal preprocessing path.

Debug-only instrumentation gated by SGLANG_DEBUG_MM_TIMING. When enabled, each
stage records elapsed wall time (GPU stages synchronize before and after so
async kernel queues do not misattribute time), and one JSON line per request
is emitted at INFO level:

    [mm-timing] {"rid": ..., "load_ms": ..., "gpu_preprocess_ms": ..., ...}

The synchronize calls serialize the GPU pipeline, so enable this only for
measurement runs, never in production serving.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from typing import Dict, Iterator, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_local = threading.local()


def enabled() -> bool:
    return envs.SGLANG_DEBUG_MM_TIMING.get()


def _records() -> Optional[Dict[str, float]]:
    return getattr(_local, "records", None)


@contextlib.contextmanager
def request_scope(rid: str) -> Iterator[None]:
    """Collect stage timings for one request and log them on exit.

    Only the outermost scope on a thread logs; stages recorded from worker
    threads (io/processor executors) fall back to per-stage logging because
    they cannot see the event-loop thread's scope.
    """
    if not enabled():
        yield
        return
    if _records() is not None:  # nested scope: let the outer one report
        yield
        return
    _local.records = {}
    start = time.perf_counter()
    try:
        yield
    finally:
        records = _local.records
        _local.records = None
        records["total_ms"] = round((time.perf_counter() - start) * 1e3, 2)
        logger.info("[mm-timing] %s", json.dumps({"rid": rid, **records}))


@contextlib.contextmanager
def stage(name: str, *, gpu_sync: bool = False) -> Iterator[None]:
    """Time one stage. gpu_sync=True brackets the stage with cuda synchronize
    so queued async work is attributed to the stage that launched it."""
    if not enabled():
        yield
        return
    if gpu_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if gpu_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = round((time.perf_counter() - start) * 1e3, 2)
        records = _records()
        if records is not None:
            key = f"{name}_ms"
            records[key] = round(records.get(key, 0.0) + elapsed_ms, 2)
        else:
            # Worker thread without a request scope: log the stage directly.
            logger.info(
                "[mm-timing] %s", json.dumps({"stage": name, "ms": elapsed_ms})
            )
