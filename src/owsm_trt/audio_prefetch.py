"""Bounded CPU audio loading; feature extraction stays on the caller's CUDA stream."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch

from .audio import AudioLike


@contextmanager
def prefetch_batches(
    batches: Iterable[list[AudioLike]], load: Callable[[AudioLike], torch.Tensor], workers: int = 0,
) -> Iterator[Iterator[list[torch.Tensor]]]:
    """Yield batches in order, with at most two submitted batches including the current one.

    The caller must exit this context on early termination so pending reads are cancelled and
    running reads joined. Only independent file handles may be loaded by the worker threads.
    CPU autocast stays synchronous because its thread-local state affects resampling.
    """
    if type(workers) is not int or workers not in (0, 1, 2):
        raise ValueError("audio prefetch workers must be 0, 1 or 2")
    if not workers or torch.is_autocast_enabled("cpu"):
        yield ([load(item) for item in batch] for batch in batches)
        return

    @torch.inference_mode()
    def read(batch):
        return [load(item) for item in batch]

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="owsm-audio")
    pending = deque()
    source = iter(batches)

    def submit():
        batch = next(source, None)
        if batch is not None:
            pending.append(executor.submit(read, batch))

    def ordered():
        submit()
        submit()
        while pending:
            future = pending.popleft()
            result = future.result()  # Observe read failures in input order.
            del future
            yield result
            del result
            submit()

    output = ordered()
    try:
        yield output
    finally:
        output.close()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
