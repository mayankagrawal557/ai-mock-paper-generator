import os
import re
import uuid
import logging
from typing import List
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Rate Limiting & Security
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Celery Task Integration
from celery.result import AsyncResult
from celery_worker import celery_app, generate_mock_paper_task
from ai_engine import get_gemini_api_key

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EnterpriseMockPaperApp")

# Initialize SlowAPI Limiter (IP-based)
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# Initialize FastAPI application
app = FastAPI(
    title="AI Mock Paper Generator (Enterprise RAG)",
    description="Production-grade asynchronous Exam Paper Setter powered by Celery, Redis, Qdrant, and Gemini",
    version="2.0.0"
)

# Attach rate limiter to application state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static mounts
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Production Security & File Constraints
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB strict limit per file
PDF_MAGIC_SIGNATURE = b"%PDF-"
STAGING_BASE_DIR = os.getenv("STAGING_DIR", os.path.join(os.path.expanduser("~"), ".mock_paper_staging"))
os.makedirs(STAGING_BASE_DIR, exist_ok=True)


# In-memory background task registry (graceful fallback when Redis is offline)
IN_MEMORY_JOBS: dict[str, dict[str, Any]] = {}


def execute_in_memory_task(
    task_id: str,
    task_dir: str,
    syllabus_filenames: List[str],
    pyq_filenames: List[str],
    exam_name: str,
    total_marks: int,
    duration_minutes: int,
    difficulty: str,
    include_solutions: bool,
    model_name: str
):
    """Fallback async worker when running on standalone hosts without Redis."""
    from ai_engine import run_rag_pipeline
    import shutil

    def progress_callback(stage: str, percentage: int):
        IN_MEMORY_JOBS[task_id] = {
            "task_id": task_id,
            "status": "PROGRESS",
            "percentage": percentage,
            "stage": stage
        }

    try:
        progress_callback("Loading documents from staging area...", 10)
        syllabus_files = []
        for fname in syllabus_filenames:
            p = os.path.join(task_dir, "syllabus", fname)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    syllabus_files.append((fname, f.read()))

        pyq_files = []
        for fname in pyq_filenames:
            p = os.path.join(task_dir, "pyq", fname)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    pyq_files.append((fname, f.read()))

        result = run_rag_pipeline(
            syllabus_files=syllabus_files,
            pyq_files=pyq_files,
            exam_name=exam_name,
            total_marks=total_marks,
            duration_minutes=duration_minutes,
            difficulty=difficulty,
            include_solutions=include_solutions,
            model_name=model_name,
            progress_callback=progress_callback
        )
        IN_MEMORY_JOBS[task_id] = {
            "task_id": task_id,
            "status": "SUCCESS",
            "percentage": 100,
            "stage": "Completed",
            "result": result
        }
    except Exception as exc:
        logger.error(f"In-memory task {task_id} failed: {exc}", exc_info=True)
        IN_MEMORY_JOBS[task_id] = {
            "task_id": task_id,
            "status": "FAILURE",
            "percentage": 0,
            "error": str(exc)
        }
    finally:
        try:
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
        except Exception:
            pass


def has_valid_api_key() -> bool:
    """Check if a real Gemini API key is configured."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return bool(key and key.strip() and key.strip() != "your_gemini_api_key_here")


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal vulnerabilities."""
    cleaned = Path(filename).name
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", cleaned)
    return cleaned or "document.pdf"


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Render the single-page dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"api_key_configured": has_valid_api_key()}
    )


@app.get("/health")
async def health_check():
    """Enterprise Health Check monitoring API key, Redis, and Celery status."""
    redis_healthy = False
    try:
        celery_app.connection().connect()
        redis_healthy = True
    except Exception:
        redis_healthy = False

    return {
        "status": "healthy" if (has_valid_api_key() and redis_healthy) else "degraded",
        "gemini_api_key_configured": has_valid_api_key(),
        "redis_broker_connected": redis_healthy,
        "service": "AI Mock Paper Generator (Enterprise)"
    }


@app.post("/api/generate", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/minute")
async def submit_paper_generation_job(
    request: Request,
    syllabus_files: List[UploadFile] = File(..., description="One or more Syllabus PDF documents"),
    pyq_files: List[UploadFile] = File(..., description="One or more Past Year Question (PYQ) PDF documents"),
    exam_name: str = Form("Semester Final Mock Examination"),
    total_marks: int = Form(100),
    duration_minutes: int = Form(180),
    difficulty: str = Form("Medium"),
    include_solutions: bool = Form(True),
    model_name: str = Form("gemini-3.5-flash")
):
    """
    Asynchronously accept PDFs, enforce security checks, stage files,
    and dispatch generation to the Celery worker queue.
    """
    # 1. Ensure API Key exists
    try:
        get_gemini_api_key()
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ve))

    # 2. Validate non-empty file arrays
    if not syllabus_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one Syllabus PDF is required.")
    if not pyq_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one PYQ PDF is required.")

    # 3. Create isolated staging workspace for this job
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(STAGING_BASE_DIR, task_id)
    syllabus_dir = os.path.join(task_dir, "syllabus")
    pyq_dir = os.path.join(task_dir, "pyq")
    os.makedirs(syllabus_dir, exist_ok=True)
    os.makedirs(pyq_dir, exist_ok=True)

    # 4. Helper for validation and file writing
    async def validate_and_stage_files(files: List[UploadFile], target_dir: str, category: str) -> List[str]:
        saved_names = []
        for file in files:
            # Check extension
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file '{file.filename}'. Only standard PDF documents are permitted for {category}."
                )

            content = await file.read()

            # Check minimum content and magic bytes
            if len(content) < 5 or not content.startswith(PDF_MAGIC_SIGNATURE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Security alert: '{file.filename}' is corrupted or does not contain a valid PDF signature header."
                )

            # Strict 10MB limit
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{file.filename}' exceeds the maximum enterprise limit of 10MB."
                )

            safe_name = sanitize_filename(file.filename)
            dest_path = os.path.join(target_dir, safe_name)
            with open(dest_path, "wb") as f:
                f.write(content)
            saved_names.append(safe_name)

        return saved_names

    try:
        staged_syllabus = await validate_and_stage_files(syllabus_files, syllabus_dir, "Syllabus")
        staged_pyq = await validate_and_stage_files(pyq_files, pyq_dir, "PYQ")
    except Exception as e:
        # Cleanup staging on early validation failure
        if os.path.exists(task_dir):
            import shutil
            shutil.rmtree(task_dir, ignore_errors=True)
        raise e

    # 5. Dispatch async background task (Celery or In-Memory fallback)
    try:
        celery_app.connection().connect()
        generate_mock_paper_task.apply_async(
            kwargs={
                "task_dir": task_dir,
                "syllabus_filenames": staged_syllabus,
                "pyq_filenames": staged_pyq,
                "exam_name": exam_name.strip() or "Mock Examination",
                "total_marks": total_marks if total_marks > 0 else 100,
                "duration_minutes": duration_minutes if duration_minutes > 0 else 180,
                "difficulty": difficulty,
                "include_solutions": include_solutions,
                "model_name": model_name
            },
            task_id=task_id
        )
        logger.info(f"Queued Celery generation task with ID: {task_id}")
    except Exception as dispatch_err:
        logger.warning(f"Redis/Celery broker offline ({dispatch_err}). Falling back to native in-memory worker.")
        import threading
        IN_MEMORY_JOBS[task_id] = {
            "task_id": task_id,
            "status": "PROGRESS",
            "percentage": 5,
            "stage": "Job queued in native async worker..."
        }
        worker_thread = threading.Thread(
            target=execute_in_memory_task,
            args=(
                task_id,
                task_dir,
                staged_syllabus,
                staged_pyq,
                exam_name.strip() or "Mock Examination",
                total_marks if total_marks > 0 else 100,
                duration_minutes if duration_minutes > 0 else 180,
                difficulty,
                include_solutions,
                model_name
            ),
            daemon=True
        )
        worker_thread.start()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "success": True,
            "task_id": task_id,
            "status": "QUEUED",
            "message": "Paper generation task successfully scheduled in background worker queue."
        }
    )


@app.get("/api/status/{task_id}")
async def check_generation_task_status(task_id: str):
    """
    Polling endpoint: Returns the real-time execution state of a queued Celery or In-Memory job.
    """
    # 1. Check in-memory job registry first (fastest & handles standalone Render web service)
    if task_id in IN_MEMORY_JOBS:
        return IN_MEMORY_JOBS[task_id]

    # 2. Check Celery task result from Redis
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        state = task_result.state
    except Exception as exc:
        logger.warning(f"Unable to query task status from broker: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "task_id": task_id,
                "status": "UNAVAILABLE",
                "error": "Task status store (Redis) is currently unreachable. Ensure Docker services are running."
            }
        )

    if state == "PENDING":
        return {
            "task_id": task_id,
            "status": "PENDING",
            "percentage": 5,
            "stage": "Waiting in queue for an available Celery worker..."
        }

    elif task_result.state == "PROGRESS":
        info = task_result.info or {}
        return {
            "task_id": task_id,
            "status": "PROGRESS",
            "percentage": info.get("percentage", 30),
            "stage": info.get("stage", "Processing RAG pipeline...")
        }

    elif task_result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "percentage": 100,
            "stage": "Completed",
            "result": task_result.result
        }

    elif task_result.state == "FAILURE":
        info = task_result.info or {}
        err_msg = str(task_result.result) if task_result.result else info.get("error", "Unknown worker error")
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "percentage": 0,
            "error": err_msg
        }

    return {
        "task_id": task_id,
        "status": task_result.state,
        "percentage": 10,
        "stage": f"Job state: {task_result.state}"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
