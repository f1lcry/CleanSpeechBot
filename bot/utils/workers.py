"""Asyncio-based task queue management."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("bot.workers")


class TaskQueueManager:
    """Manage an ``asyncio.Queue`` and worker lifecycle for background tasks."""

    def __init__(self, maxsize: int) -> None:
        """Initialize the task queue with the specified maximum size."""

        self.queue: asyncio.Queue[Callable[[], Awaitable[Any]]] = asyncio.Queue(maxsize=maxsize)
        self._workers: list[asyncio.Task[None]] = []
        self._shutdown_event = asyncio.Event()

    async def start_workers(self, count: int) -> None:
        """Start the configured number of worker tasks.

        TODO: Extend to support error backoff and metrics.
        """

        for index in range(count):
            worker = asyncio.create_task(self._worker_loop(index), name=f"task-worker-{index}")
            self._workers.append(worker)
            logger.debug("Started worker %s", worker.get_name())

    async def enqueue(self, task_factory: Callable[[], Awaitable[Any]]) -> None:
        """Put a new task factory into the queue for processing."""

        await self.queue.put(task_factory)
        logger.debug("Task enqueued. Queue size: %s", self.queue.qsize())

    async def _worker_loop(self, worker_id: int) -> None:
        """Continuously process tasks from the queue until shutdown is triggered."""

        while not self._shutdown_event.is_set():
            try:
                task_factory = await self.queue.get()
                logger.info("Worker %s processing task", worker_id)
                await self._execute_task(task_factory)
            except asyncio.CancelledError:
                logger.info("Worker %s received cancellation.", worker_id)
                break
            except Exception as exc:  # TODO: narrow down exception handling when logic is implemented.
                logger.exception("Worker %s encountered an error: %s", worker_id, exc)
            finally:
                self.queue.task_done()

    async def _execute_task(self, task_factory: Callable[[], Awaitable[Any]]) -> None:
        """Execute a single task produced by the provided factory."""

        task = task_factory()
        await task

    async def shutdown(self) -> None:
        """Request graceful shutdown of workers and wait for completion."""

        logger.info("Shutdown initiated for task queue manager.")
        self._shutdown_event.set()

        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        logger.info("All workers have been shut down.")
