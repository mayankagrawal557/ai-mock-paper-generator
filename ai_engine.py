import os
import io
import uuid
import logging
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")


def get_gemini_api_key() -> str:
    """Retrieve and validate the Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or invalid. "
            "Please configure GEMINI_API_KEY in your .env file or environment variables."
        )
    return api_key.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> str:
    """Extract clean text content from PDF file bytes using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages)
        if not full_text.strip():
            raise ValueError(
                f"No selectable text found in '{filename}'. "
                "The PDF might be scanned/image-only or password-protected."
            )
        return full_text
    except Exception as exc:
        logger.error(f"Failed to extract text from {filename}: {exc}")
        raise ValueError(f"Error reading PDF '{filename}': {str(exc)}")


def process_uploaded_documents(
    syllabus_files: List[Tuple[str, bytes]],
    pyq_files: List[Tuple[str, bytes]]
) -> List[Document]:
    """
    Parse uploaded Syllabus and PYQ files into chunked LangChain Document objects
    tagged with metadata (source filename and document type).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_docs: List[Document] = []

    # 1. Process Syllabus PDFs
    for filename, file_bytes in syllabus_files:
        text = extract_text_from_pdf_bytes(file_bytes, filename)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": filename, "doc_type": "syllabus"}
                )
            )
        logger.info(f"Processed Syllabus '{filename}': {len(chunks)} chunks.")

    # 2. Process PYQ PDFs
    for filename, file_bytes in pyq_files:
        text = extract_text_from_pdf_bytes(file_bytes, filename)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": filename, "doc_type": "pyq"}
                )
            )
        logger.info(f"Processed PYQ '{filename}': {len(chunks)} chunks.")

    return all_docs


def get_qdrant_client() -> QdrantClient:
    """
    Initialize Qdrant client connection.
    Connects to Docker Qdrant service via QDRANT_URL or falls back to in-memory mode.
    """
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        logger.info(f"Connecting to Qdrant cluster at {qdrant_url}")
        return QdrantClient(url=qdrant_url)
    
    # Fallback to ephemeral in-memory Qdrant client for zero-dependency local dev
    logger.info("QDRANT_URL not set; using high-speed in-memory Qdrant client.")
    return QdrantClient(":memory:")


def build_qdrant_vectorstore(
    documents: List[Document],
    api_key: str
) -> Tuple[QdrantClient, str, GoogleGenerativeAIEmbeddings]:
    """
    Index document chunks into a dedicated, isolated Qdrant collection.
    Uses Google's gemini-embedding-001 (vector size 3072, cosine distance).
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    client = get_qdrant_client()
    collection_name = f"exam_rag_{uuid.uuid4().hex[:12]}"

    # Embed all document chunks
    texts = [doc.page_content for doc in documents]
    logger.info(f"Generating embeddings for {len(texts)} document chunks...")
    vectors = embeddings.embed_documents(texts)
    vector_size = len(vectors[0])

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE
        )
    )

    # Ingest points into Qdrant
    points = []
    for idx, (doc, vec) in enumerate(zip(documents, vectors)):
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "page_content": doc.page_content,
                    "doc_type": doc.metadata.get("doc_type", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Successfully indexed {len(points)} points into Qdrant collection '{collection_name}'.")

    return client, collection_name, embeddings


def retrieve_rag_context_from_qdrant(
    client: QdrantClient,
    collection_name: str,
    embeddings: GoogleGenerativeAIEmbeddings
) -> Tuple[str, str]:
    """
    Query Qdrant collection with payload filters for syllabus boundaries and PYQ patterns.
    """
    syllabus_queries = [
        "syllabus core units modules topics subtopics course outcomes",
        "detailed topics concepts weightage list of chapters"
    ]

    pyq_queries = [
        "exam instructions section breakdown total marks time allowed question paper format",
        "question types numerical problems multiple choice short answer long answer derivation case study",
        "marking scheme difficulty level bloom taxonomy problem statements"
    ]

    def query_docs(query_text: str, doc_type: str, limit: int = 4) -> List[str]:
        query_vector = embeddings.embed_query(query_text)
        search_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_type",
                    match=qmodels.MatchValue(value=doc_type)
                )
            ]
        )
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )
        results = []
        for point in search_results:
            source = point.payload.get("source", doc_type)
            content = point.payload.get("page_content", "")
            formatted = f"[{source}]:\n{content}"
            if formatted not in results:
                results.append(formatted)
        return results

    # Retrieve syllabus chunks
    syllabus_chunks: List[str] = []
    for q in syllabus_queries:
        for chunk in query_docs(q, "syllabus", limit=4):
            if chunk not in syllabus_chunks:
                syllabus_chunks.append(chunk)

    # Retrieve PYQ chunks
    pyq_chunks: List[str] = []
    for q in pyq_queries:
        for chunk in query_docs(q, "pyq", limit=5):
            if chunk not in pyq_chunks:
                pyq_chunks.append(chunk)

    syllabus_context = "\n\n---\n\n".join(syllabus_chunks) if syllabus_chunks else "No specific syllabus chunks found."
    pyq_context = "\n\n---\n\n".join(pyq_chunks) if pyq_chunks else "No specific PYQ chunks found."

    return syllabus_context, pyq_context


def parse_llm_content(response: Any) -> str:
    """Safely extracts text string from string, list-of-dicts, or structured response."""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return "\n".join(parts)
    return str(response)


def generate_mock_paper(
    syllabus_context: str,
    pyq_context: str,
    exam_name: str,
    total_marks: int,
    duration_minutes: int,
    difficulty: str,
    include_solutions: bool,
    model_name: str,
    api_key: str
) -> str:
    """
    Execute Gemini LLM with tailored instructions to generate a balanced,
    high-probability mock exam paper grounded in syllabus and PYQs.
    """
    valid_models = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-pro-latest"]
    chosen_model = model_name if model_name in valid_models else "gemini-3.5-flash"

    system_instruction = (
        "You are an expert University Examination Board Chair and Master Paper Setter. "
        "Your task is to craft a completely original, highly realistic Mock Examination Paper "
        "based strictly on the provided Syllabus and past examination patterns (PYQs)."
    )

    solution_prompt_part = ""
    if include_solutions:
        solution_prompt_part = """
### SECTION: ANSWER KEY & MARKING SCHEME
At the very end of the question paper, provide:
1. Detailed Answer Key & Step-by-Step Solutions for numerical/problem questions.
2. Key points / Rubric expected for conceptual and long-answer questions.
3. Mark breakdown for partial credit.
"""

    prompt = f"""{system_instruction}

---
### EXAMINATION SPECIFICATIONS:
- **Course / Examination Title**: {exam_name}
- **Total Marks**: {total_marks}
- **Duration**: {duration_minutes} minutes
- **Target Difficulty Level**: {difficulty} (Follow standard PYQ distribution: ~40% Easy/Foundational, ~40% Moderate/Application, ~20% Challenging/Analytical)

---
### CONTEXT 1: EXTRACTED SYLLABUS (MANDATORY BOUNDARY)
Only ask questions on topics that fall within this syllabus. Do NOT ask out-of-syllabus questions:
{syllabus_context}

---
### CONTEXT 2: EXTRACTED PREVIOUS YEAR QUESTION (PYQ) PATTERNS
Analyze the format, section division, question phrasing style, marks distribution, and recurring themes below:
{pyq_context}

---
### STRICT GENERATION RULES:
1. **Structural Fidelity**:
   - Replicate the exact structural pattern observed in the PYQs (e.g., Section A: Objective/Short Questions, Section B: Medium Analytical, Section C: Comprehensive/Long/Numerical).
   - Provide standard exam instructions at the top (e.g., "Answer all questions from Section A", "Assume suitable data if necessary", etc.).
2. **Topic Alignment & Weightage**:
   - Ensure proportional coverage across the modules/units present in the syllabus.
   - Do NOT introduce any topic not covered in the syllabus context.
3. **Question Novelty**:
   - **DO NOT** simply copy-paste questions word-for-word from the PYQ context.
   - Synthesize fresh, realistic, and intellectually rigorous questions that test identical concepts, difficulty, and depth.
4. **Marking Distribution**:
   - Assign clear marks for each question and sub-part (e.g., `[5 Marks]`, `[2 + 3 = 5 Marks]`).
   - The sum of compulsory questions must equal exactly {total_marks} marks (clearly specify internal choices like "Answer either Q3(a) or Q3(b)" if typical for this exam pattern).
5. **Output Formatting**:
   - Format cleanly in professional GitHub-Flavored Markdown.
   - Include a formal Exam Header (Institution / Exam Name, Time Allowed, Max Marks, Date, Instructions).
   - Use clear markdown headings (`## Section A`, `### Question 1`, etc.).
   - Use standard LaTeX notation (enclosed in `$`) for mathematical formulas, chemical equations, or scientific symbols if applicable.

{solution_prompt_part}

Generate the complete, flawless Mock Exam Paper now:
"""

    models_to_try = [chosen_model]
    if chosen_model != "gemini-3.5-flash":
        models_to_try.append("gemini-3.5-flash")
    if "gemini-flash-latest" not in models_to_try:
        models_to_try.append("gemini-flash-latest")

    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"Invoking Gemini model '{model}'...")
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.35,
                max_output_tokens=8192
            )
            response = llm.invoke(prompt)
            return parse_llm_content(response)
        except Exception as e:
            logger.warning(f"Model '{model}' failed with: {e}. Trying fallback if available.")
            last_error = e

    raise last_error


def run_rag_pipeline(
    syllabus_files: List[Tuple[str, bytes]],
    pyq_files: List[Tuple[str, bytes]],
    exam_name: str = "Standard Mock Examination",
    total_marks: int = 100,
    duration_minutes: int = 180,
    difficulty: str = "Medium",
    include_solutions: bool = True,
    model_name: str = "gemini-3.5-flash",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Enterprise RAG pipeline orchestrator with Qdrant and progress reporting.
    """
    api_key = get_gemini_api_key()

    if not syllabus_files:
        raise ValueError("At least one Syllabus PDF file must be provided.")
    if not pyq_files:
        raise ValueError("At least one Previous Year Question (PYQ) PDF file must be provided.")

    if progress_callback:
        progress_callback("Parsing and chunking PDF documents...", 20)

    logger.info("Step 1: Parsing and chunking PDF documents...")
    documents = process_uploaded_documents(syllabus_files, pyq_files)
    if not documents:
        raise ValueError("Unable to extract any text chunks from the uploaded PDFs.")

    if progress_callback:
        progress_callback("Generating embeddings & indexing into Qdrant...", 45)

    logger.info(f"Step 2: Indexing {len(documents)} chunks into Qdrant vector store...")
    client, collection_name, embeddings = build_qdrant_vectorstore(documents, api_key)

    try:
        if progress_callback:
            progress_callback("Retrieving syllabus scope & PYQ question blueprints...", 70)

        logger.info("Step 3: Retrieving contextual syllabus topics and PYQ patterns...")
        syllabus_ctx, pyq_ctx = retrieve_rag_context_from_qdrant(client, collection_name, embeddings)

        if progress_callback:
            progress_callback("Synthesizing original mock paper with Google Gemini...", 85)

        logger.info(f"Step 4: Generating Mock Paper using Gemini ({model_name})...")
        paper_markdown = generate_mock_paper(
            syllabus_context=syllabus_ctx,
            pyq_context=pyq_ctx,
            exam_name=exam_name,
            total_marks=total_marks,
            duration_minutes=duration_minutes,
            difficulty=difficulty,
            include_solutions=include_solutions,
            model_name=model_name,
            api_key=api_key
        )

        syllabus_chunk_count = sum(1 for d in documents if d.metadata.get("doc_type") == "syllabus")
        pyq_chunk_count = sum(1 for d in documents if d.metadata.get("doc_type") == "pyq")

        if progress_callback:
            progress_callback("Paper generation completed!", 100)

        return {
            "success": True,
            "exam_name": exam_name,
            "total_marks": total_marks,
            "duration_minutes": duration_minutes,
            "difficulty": difficulty,
            "model_used": model_name,
            "paper_markdown": paper_markdown,
            "stats": {
                "total_chunks_indexed": len(documents),
                "syllabus_chunks": syllabus_chunk_count,
                "pyq_chunks": pyq_chunk_count,
                "syllabus_files_count": len(syllabus_files),
                "pyq_files_count": len(pyq_files)
            }
        }
    finally:
        # Reclaim vector collection memory
        try:
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Qdrant collection '{collection_name}' safely deleted.")
        except Exception as e:
            logger.warning(f"Could not cleanly delete collection {collection_name}: {e}")
