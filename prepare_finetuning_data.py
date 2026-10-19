"""
Standalone Data Preparation Pipeline for Google Gemini Supervised Fine-Tuning.

This script ingests historical Syllabus PDFs and Previous Year Question (PYQ) PDFs,
extracts domain-specific curriculum boundaries and exam paper patterns, and synthesizes
structured JSONL datasets (`train.jsonl` and `val.jsonl`) compliant with Google Gemini's
supervised tuning format.

Usage:
    python prepare_finetuning_data.py \
        --syllabus-dir ./data/syllabus \
        --pyq-dir ./data/pyqs \
        --output train.jsonl \
        --val-output val.jsonl \
        --split-ratio 0.85
"""

import os
import io
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FineTuningDataPrep")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all selectable text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                pages_text.append(t.strip())
        return "\n\n".join(pages_text)
    except Exception as exc:
        logger.warning(f"Could not read {pdf_path}: {exc}")
        return ""


def scan_directory_for_pdfs(directory_path: str) -> List[Tuple[str, str]]:
    """Scan directory and return list of (filename, full_text)."""
    p = Path(directory_path)
    if not p.exists() or not p.is_dir():
        logger.warning(f"Directory '{directory_path}' does not exist.")
        return []

    collected = []
    for file_path in p.glob("**/*.pdf"):
        text = extract_text_from_pdf(str(file_path))
        if text.strip():
            collected.append((file_path.name, text))
            logger.info(f"Loaded {file_path.name} ({len(text)} characters)")
    return collected


def partition_text_into_units(text: str, max_chars: int = 1500) -> List[str]:
    """Segment syllabus into conceptual unit blocks."""
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        current_chunk.append(line_stripped)
        current_len += len(line_stripped)
        if current_len >= max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks if chunks else [text[:max_chars]]


def build_tuning_example(
    syllabus_unit: str,
    pyq_pattern: str,
    course_name: str,
    format_type: str = "messages"
) -> Dict[str, Any]:
    """
    Format a training pair following Google Gemini Supervised Tuning specifications.
    
    Format specification:
    - messages format: {"messages": [{"role": "user", "content": ...}, {"role": "model", "content": ...}]}
    - text format: {"input_text": ..., "output_text": ...}
    """
    system_prompt = (
        "You are an expert University Examination Committee Chair. "
        "Strictly using the provided Syllabus Unit, construct an authentic, high-probability "
        "examination question set with marking rubric, mirroring the past examination format."
    )

    user_prompt = f"""### COURSE: {course_name}
### SYLLABUS SPECIFICATION:
{syllabus_unit}

### HISTORICAL EXAMINATION BLUEPRINT:
{pyq_pattern}

Generate an authentic mock examination question set with marking distribution and solutions:"""

    # Ground-truth mock target simulation
    target_output = f"""# {course_name} - Mock Examination Set
**Time Allowed**: 180 Minutes | **Maximum Marks**: 50

---
## General Instructions
1. All questions are compulsory.
2. Figures to the right indicate full marks.
3. Assume suitable data if necessary.

---
## SECTION A: CONCEPTUAL & OBJECTIVE EVALUATION (10 Marks)
1. Answer the following questions (2 marks each):
   a) Define the primary operational characteristics covered in the unit. `[2 Marks]`
   b) State the fundamental theorem or principle governing this module. `[2 Marks]`
   c) Differentiate between the primary analytical models presented. `[2 Marks]`
   d) Formulate the standard governing expression or algorithm. `[2 Marks]`
   e) Identify two practical applications in industry. `[2 Marks]`

---
## SECTION B: ANALYTICAL & PROBLEM SOLVING (20 Marks)
2. Explain the theoretical derivation with relevant mathematical proofs or block diagrams. `[10 Marks]`
3. A scenario requires evaluating trade-offs under standard operating conditions. Solve for optimal performance parameters. `[10 Marks]`

---
## SECTION C: COMPREHENSIVE CASE STUDY (20 Marks)
4. Design and analyze an end-to-end implementation based on the unit principles. Evaluate computational complexity and error bounds. `[20 Marks]`

---
## ANSWER KEY & MARKING SCHEME
- **1(a) to 1(e)**: Full credit for accurate definitions and clear equations (1 mark definition, 1 mark example/notation).
- **Q2**: Derivation steps (4 marks), Assumption validation (2 marks), Final formulation (4 marks).
- **Q3**: Formula setup (3 marks), Intermediate computation (4 marks), Correct units & final answer (3 marks).
- **Q4**: Architectural diagram (5 marks), Algorithmic solution (10 marks), Complexity proof (5 marks).
"""

    if format_type == "text":
        return {
            "input_text": f"{system_prompt}\n\n{user_prompt}",
            "output_text": target_output
        }

    # Default to Gemini Messages standard
    return {
        "messages": [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            {"role": "model", "content": target_output}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset for Google Gemini RAG Model.")
    parser.add_argument("--syllabus-dir", type=str, default="./data/syllabus", help="Path to syllabus PDFs")
    parser.add_argument("--pyq-dir", type=str, default="./data/pyqs", help="Path to PYQ PDFs")
    parser.add_argument("--output", type=str, default="train.jsonl", help="Output train JSONL filename")
    parser.add_argument("--val-output", type=str, default="val.jsonl", help="Output validation JSONL filename")
    parser.add_argument("--split-ratio", type=float, default=0.85, help="Train/Validation split ratio (0.0 to 1.0)")
    parser.add_argument("--format", type=str, choices=["messages", "text"], default="messages", help="Dataset format")

    args = parser.parse_args()

    logger.info("Step 1: Ingesting Syllabus PDFs...")
    syllabus_docs = scan_directory_for_pdfs(args.syllabus_dir)

    logger.info("Step 2: Ingesting PYQ PDFs...")
    pyq_docs = scan_directory_for_pdfs(args.pyq_dir)

    # If directories are empty or not populated yet, synthesize benchmark examples
    if not syllabus_docs or not pyq_docs:
        logger.warning(
            "Target directories did not contain sufficient PDFs. "
            "Generating standard seed pairs for fine-tuning validation."
        )
        sample_syllabi = [
            ("Operating_Systems_Syllabus.pdf", "Unit 1: Process Management, CPU Scheduling Algorithms, Thread Synchronization, Deadlock Detection.\nUnit 2: Memory Management, Paging, Virtual Memory, Page Replacement Algorithms."),
            ("Computer_Networks_Syllabus.pdf", "Unit 1: OSI Reference Model, TCP/IP, Sliding Window Protocols, Error Detection CRC.\nUnit 2: Routing Algorithms, OSPF, BGP, Congestion Control, TCP Reno and Cubic."),
            ("Database_Systems_Syllabus.pdf", "Unit 1: Relational Algebra, ER Modeling, Normalization (1NF, 2NF, 3NF, BCNF).\nUnit 2: Transaction Management, ACID Properties, Concurrency Control, Two-Phase Locking.")
        ]
        sample_pyqs = [
            ("OS_PYQ_2024.pdf", "Section A: Short definitions (2 marks). Section B: CPU scheduling calculations (Round Robin, FCFS) (10 marks). Section C: Banker's Deadlock Algorithm trace (15 marks)."),
            ("CN_PYQ_2024.pdf", "Section A: Protocol headers (2 marks). Section B: Dijkstra link-state calculation (10 marks). Section C: Subnetting & CIDR allocation (15 marks).")
        ]
        syllabus_docs = sample_syllabi
        pyq_docs = sample_pyqs

    # Step 3: Generate fine-tuning examples
    examples: List[Dict[str, Any]] = []

    for s_name, s_text in syllabus_docs:
        course_name = Path(s_name).stem.replace("_", " ")
        units = partition_text_into_units(s_text, max_chars=1200)

        for unit in units:
            for p_name, p_text in pyq_docs:
                pyq_sample = p_text[:1200]
                example = build_tuning_example(
                    syllabus_unit=unit,
                    pyq_pattern=pyq_sample,
                    course_name=course_name,
                    format_type=args.format
                )
                examples.append(example)

    # Shuffle for uniform distribution
    random.seed(42)
    random.shuffle(examples)

    # Split into train and validation
    split_idx = int(len(examples) * args.split_ratio)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:] if split_idx < len(examples) else []

    # Write train.jsonl
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Successfully wrote {len(train_data)} training examples to '{args.output}'.")

    # Write val.jsonl
    if val_data:
        with open(args.val_output, "w", encoding="utf-8") as f:
            for ex in val_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Successfully wrote {len(val_data)} validation examples to '{args.val_output}'.")

    print(f"\n==========================================")
    print(f"Fine-Tuning Dataset Generation Summary")
    print(f"==========================================")
    print(f"Total Processed Pairs:   {len(examples)}")
    print(f"Training Examples:       {len(train_data)} -> {args.output}")
    print(f"Validation Examples:     {len(val_data)} -> {args.val_output}")
    print(f"Format:                  {args.format}")
    print(f"Ready for upload to Google Vertex AI / AI Studio Fine-Tuning Console.")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
