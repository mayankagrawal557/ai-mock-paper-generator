# 🎓 AI Mock Paper Generator (Enterprise Production Edition)

A high-concurrency, asynchronous, Retrieval-Augmented Generation (RAG) platform that ingests an academic **Syllabus PDF** and **Previous Year Questions (PYQs) PDFs** to generate an original, balanced, and exam-ready mock paper.

Engineered to handle **100+ concurrent requests** using **FastAPI**, **Celery**, **Redis**, **Qdrant Vector Database**, **Google Gemini**, and **SlowAPI Rate Limiting**.

---

## 🏛️ Enterprise Architecture Overview

```mermaid
flowchart TD
    Client["Client / Web UI (Tailwind CSS)"] -->|1. POST /api/generate| API["FastAPI Gateway (SlowAPI Protected)"]
    API -->|2. Validate PDF Header & Size| Staging["Staging Disk Storage"]
    API -->|3. Dispatch Task ID| RedisQueue["Redis Message Broker"]
    API -->|4. Return 202 Accepted| Client

    Worker["Celery Worker Cluster (concurrency=4)"] -->|5. Fetch Job| RedisQueue
    Worker -->|6. Ingest PDFs from Staging| DocProcessor["pypdf Chunking Engine"]
    DocProcessor -->|7. Embed Chunks (gemini-embedding-001)| Qdrant[("Qdrant Vector DB (Docker)")]
    Worker -->|8. Query Blueprint & Boundaries| Qdrant
    Worker -->|9. Synthesize Mock Paper| Gemini["Google Gemini API (3.5 / 3.8 Flash)"]
    Worker -->|10. Write Result & Progress (0-100%)| RedisBackend["Redis Result Backend"]
    Worker -->|11. Reclaim Vector Collection & Staging| Cleanup["Secure Cleanup"]

    Client -->|12. Poll GET /api/status/:id (1.5s)| API
    API -->|13. Read State| RedisBackend
    API -->|14. Return Markdown & Stats| Client
```

---

## 🚀 Key Upgrades in Version 2.0 (Enterprise)

1. **Background Task Queue (Celery + Redis)**:
   - FastAPI no longer blocks on document chunking, vector ingestion, or LLM generation.
   - Long-running RAG pipelines are executed by independent Celery workers with real-time percentage progress updates (`PENDING` -> `PROGRESS` -> `SUCCESS` / `FAILURE`).
2. **Concurrency-Safe Vector Storage (Qdrant)**:
   - Migrated from local ChromaDB (which locks SQLite under concurrent write load) to **Qdrant**, a high-performance vector search database written in Rust.
   - Supports multi-tenant payload filtering (`doc_type: syllabus` vs `pyq`), isolated collections, and ephemeral garbage collection.
3. **Security & Rate Limiting (SlowAPI)**:
   - Integrated `SlowAPI` with IP-based rate limiting (5 requests/minute per client on generation).
   - Strict file validation: Magic signature check (`%PDF-`), file size limits (<= 10MB per PDF), and path traversal sanitization.
4. **Google Gemini Supervised Fine-Tuning Pipeline**:
   - Added `prepare_finetuning_data.py`: A standalone utility that parses historical syllabus modules and past papers into structured `train.jsonl` and `val.jsonl` datasets following Gemini's supervised fine-tuning specification.
5. **Full Docker Containerization**:
   - `docker-compose.yml` spins up the entire enterprise stack with a single command:
     - `web`: FastAPI ASGI application
     - `worker`: Celery worker cluster
     - `redis`: Redis 7 broker & state backend
     - `qdrant`: Qdrant vector database

---

## 📂 Project Structure

```
AI PAPER SETTER/
├── docker-compose.yml           # Orchestration: FastAPI + Worker + Redis + Qdrant
├── Dockerfile                   # Multi-stage production container definition
├── app.py                       # FastAPI server, SlowAPI rate limiter, async endpoints
├── celery_worker.py             # Background task runner & staging cleanup
├── ai_engine.py                 # Qdrant vector store, embeddings, and Gemini synthesis
├── prepare_finetuning_data.py   # Standalone Gemini fine-tuning dataset builder
├── requirements.txt             # Production dependencies
├── .env.example                 # Configuration template
├── templates/
│   └── index.html               # Modern UI with asynchronous task polling & PDF export
└── README.md                    # System documentation
```

---

## 🐳 Quickstart with Docker Compose (Recommended)

### 1. Configure Environment
Create `.env` from template and add your Google Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Launch All Services
```bash
docker-compose up --build -d
```

### 3. Check Status
Verify all 4 microservices are healthy:
```bash
docker-compose ps
```

Visit the dashboard in your browser:
👉 **`http://localhost:8000`**

To monitor Celery worker logs:
```bash
docker-compose logs -f worker
```

---

## 💻 Manual Local Setup (Without Docker)

If you prefer to run services individually on your local system:

### 1. Start Redis
```bash
# Via Redis CLI or Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### 2. Activate Virtual Environment & Install Dependencies
```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Celery Worker
```bash
celery -A celery_worker.celery_app worker --loglevel=info --concurrency=4
```

### 4. Start FastAPI Web Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🎯 Fine-Tuning Data Preparation Script

To fine-tune Google Gemini on your institution's specific syllabi and historical question paper styles:

```bash
python prepare_finetuning_data.py \
    --syllabus-dir ./data/syllabus \
    --pyq-dir ./data/pyqs \
    --output train.jsonl \
    --val-output val.jsonl \
    --split-ratio 0.85 \
    --format messages
```

Output:
- `train.jsonl`: Formatted training examples in Google Gemini supervised tuning format.
- `val.jsonl`: Validation examples to measure convergence.
- Ready to upload directly to [Google Vertex AI](https://cloud.google.com/vertex-ai) or [Google AI Studio](https://aistudio.google.com/).

---

## 🛡️ Security & Performance Specifications

| Parameter | Configuration | Purpose |
| :--- | :--- | :--- |
| **Max File Size** | 10 MB per PDF | Prevents memory exhaustion attacks |
| **File Verification** | `%PDF-` Magic signature | Blocks disguised executable uploads |
| **Rate Limiter** | 5 requests / min (`slowapi`) | Prevents abuse & API quota drainage |
| **Worker Concurrency**| 4 parallel worker threads | Scales across CPU cores |
| **Vector Engine** | Qdrant (Cosine similarity) | Non-blocking, isolated collections |
| **Result Cache** | Redis (TTL: 3600 seconds) | Instant status checking for frontends |
