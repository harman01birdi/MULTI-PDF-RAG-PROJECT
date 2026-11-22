from celery import Celery

from config import REDIS_URL


celery_app=Celery(
    "rag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,  #adds a STARTED state so we can report "processing"
    result_expires=24 * 3600,
    task_acks_late=True,  #ack after the task finishes : a crashed workers job
    worker_prefetch_multiplier=1,   #one long job at a time per process
    broker_connection_retry_on_startup=True,
)