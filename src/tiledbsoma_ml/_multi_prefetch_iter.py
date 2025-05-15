from __future__ import annotations

import queue
from typing import Iterator, TypeVar
from concurrent import futures

_T = TypeVar("_T")

class MultiPrefetchIterator(Iterator[_T]):
    """
    Prefetches multiple items from the given iterator concurrently using a thread pool.
    """

    def __init__(
        self,
        iterator: Iterator[_T],
        prefetch: int = 4,
        pool: futures.Executor | None = None,
    ):
        self.iterator = iterator
        self.prefetch = prefetch
        self._pool = pool or futures.ThreadPoolExecutor(max_workers=prefetch)
        self._own_pool = pool is None
        self._queue: queue.Queue[_T | StopIteration | BaseException] = queue.Queue(maxsize=prefetch)
        self._stopped = False

        # Launch initial prefetching
        for _ in range(self.prefetch):
            self._submit_next()

    def _submit_next(self):
        if self._stopped:
            return
        def fetch():
            try:
                item = next(self.iterator)
                self._queue.put(item)
            except StopIteration:
                self._queue.put(StopIteration())
                self._stopped = True
            except Exception as e:
                self._queue.put(e)
                self._stopped = True
        self._pool.submit(fetch)

    def __next__(self) -> _T:
        item = self._queue.get()
        if isinstance(item, StopIteration):
            self._cleanup()
            raise StopIteration
        elif isinstance(item, Exception):
            self._cleanup()
            raise item
        else:
            # Prefetch next item to keep buffer full
            if not self._stopped:
                self._submit_next()
            return item

    def _cleanup(self):
        if self._own_pool:
            self._pool.shutdown(wait=True)

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()
