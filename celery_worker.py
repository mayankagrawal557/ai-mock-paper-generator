import os
import shutil
import logging
from typing import Dict, Any, List
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CeleryWorker")

# Redis broker and backend configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "mock_paper_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Cache results for 1 hour
    task_track_started=True,
    worker_prefetch_multiplier=1,  # Fair distribution among workers
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", 4))
)


@celery_app.task(bind=True, name="tasks.generate_mock_paper_task")
def generate_mock_paper_task(
    self,
    task_dir: str,
    syllabus_filenames: List[str],
    pyq_filenames: List[str],
    exam_name: str,
    total_marks: int,
    duration_minutes: int,
    difficulty: str,
    include_solutions: bool,
    model_name: str,
    blooms_ratio: dict = None
) -> Dict[str, Any]:
    """
    Background worker task to execute RAG paper generation without blocking FastAPI.
    Tracks state transitions and cleans up temporary file artifacts.
    """
    from ai_engine import run_rag_pipeline

    def progress_reporter(stage_desc: str, percentage: int):
        logger.info(f"Task [{self.request.id}] Progress ({percentage}%): {stage_desc}")
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": stage_desc,
                "percentage": percentage,
                "exam_name": exam_name
            }
        )

    try:
        progress_reporter("Loading uploaded PDF documents from staging area...", 10)

        # Read syllabus files from task disk staging
        syllabus_files = []
        for fname in syllabus_filenames:
            fpath = os.path.join(task_dir, "syllabus", fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    syllabus_files.append((fname, f.read()))

        # Read PYQ files from task disk staging
        pyq_files = []
        for fname in pyq_filenames:
            fpath = os.path.join(task_dir, "pyq", fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    pyq_files.append((fname, f.read()))

        if not syllabus_files or not pyq_files:
            raise ValueError("Staged PDF files could not be found or read by the Celery worker.")

        # Run enterprise RAG pipeline
        result = run_rag_pipeline(
            syllabus_files=syllabus_files,
            pyq_files=pyq_files,
            exam_name=exam_name,
            total_marks=total_marks,
            duration_minutes=duration_minutes,
            difficulty=difficulty,
            include_solutions=include_solutions,
            model_name=model_name,
            blooms_ratio=blooms_ratio,
            progress_callback=progress_reporter
        )

        return result

    except Exception as exc:
        logger.error(f"Task [{self.request.id}] failed: {exc}", exc_info=True)
        # Update state with error
        self.update_state(
            state="FAILURE",
            meta={"error": str(exc), "stage": "Failed during generation"}
        )
        raise exc

    finally:
        # Secure cleanup: Remove uploaded temporary PDFs from staging
        try:
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
                logger.info(f"Cleaned up staging folder: {task_dir}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to remove staging folder {task_dir}: {cleanup_err}")
