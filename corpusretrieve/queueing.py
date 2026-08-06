from __future__ import annotations

from fastapi import BackgroundTasks

from corpusretrieve.config import get_settings


def enqueue_job(job_id: str, background_tasks: BackgroundTasks) -> None:
    settings = get_settings()
    if settings.job_mode == "rq":
        from redis import Redis
        from rq import Queue

        queue = Queue("corpusretrieve", connection=Redis.from_url(settings.redis_url))
        queue.enqueue(
            "corpusretrieve.worker.process_job",
            job_id,
            job_timeout="2h",
            result_ttl=settings.job_ttl_seconds,
            failure_ttl=settings.job_ttl_seconds,
        )
        return

    from corpusretrieve.worker import process_job
    background_tasks.add_task(process_job, job_id)
