# Context Handover: Course Catalog PDF Processing & Model Comparison Pipeline

## Project Overview & Objectives
- **Context:** Internship at a research center evaluating automated document summarization workflows.
- **Task:** Process a batch of **168 course PDFs** stored in nested category subfolders (e.g., `INDIGO courses/Artificial Intelligence/...`), match them to a master JSON metadata file (`catalog_cards.json`), extract structured attributes, generate 150-word marketing brochure summaries using LLMs, and export everything into a single consolidated CSV dataset.
- **Research Goal:** Benchmark performance across local open-source vs. cloud-hosted LLM models by collecting performance metrics (extraction/OCR speed and LLM inference latency per model).

---

## Technical & Cost Constraints
1. **Budget Cap:** Hard limit of **RM 50 (~$11–$12 USD)**. The pipeline must avoid paid OCR or vision API calls to keep expenses well within budget.
2. **Local Environment:**
   - **OS:** Windows
   - **OCR Engine:** Tesseract OCR `v5.5.3` configured locally (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
   - **Environment Management:** API keys (`OPENROUTER_API_KEY`) and URLs are loaded dynamically via `python-dotenv`.

---

## Data Extraction & Parsing Strategy

### 1. Hybrid Hybrid Text/OCR Extraction
- **PyMuPDF (`fitz`):** Extracts raw native PDF text streams first.
- **Tesseract Fallback:** Automatically falls back to OCR rendering via PIL/PyMuPDF if extracted page text is fewer than 50 characters (handles scanned or image-only PDFs).

### 2. Precise Boundary-Based Slicing (Regex Anchoring)
To extract the **`original_overview`** cleanly—excluding company headers, contact footers, and subsequent sections—a regex boundary strategy is implemented:
- **Start Anchors:** Matches variations like `Overview`, `Program Overview`, `Course Overview`, `Why Matters`, `About this Course`.
- **End Anchors:** Truncates immediately upon encountering section titles like `Learning Objectives`, `Course Objectives`, `Program Outline`, `Course Schedule`, `Pre-requisite`, or `Learning Outcomes`.

### 3. Metadata Matching & Classification
- **Fuzzy Catalog Matching:** Uses `rapidfuzz` (`token_sort_ratio`) to match PDF filenames against titles in `catalog_cards.json`.
- **Category Classification:** Extracts the parent directory name (e.g., `Artificial Intelligence`) to preserve folder taxonomy in the dataset.

---

## Benchmark Models Evaluated
1. **Local Model:** `ollama/llama3.2:3b` (Runs locally via Ollama to eliminate API cost).
2. **Cloud Free Model:** `inclusionai/ling-3.0-flash-vl:free` (Evaluated via OpenRouter).
3. **Cloud High-Performance Model:** `meta-llama/llama-3.3-70b-instruct` (Evaluated via OpenRouter).

---

## Dataset Output Schema (`research_course_summaries.csv`)
- `id`: Unique course ID matched from `catalog_cards.json`
- `category`: Subfolder name (e.g., *Artificial Intelligence*)
- `title`: Matched course title
- `original_overview`: Cleanly extracted overview text (bounded by regex)
- `outline_summary`: Extracted schedule/agenda modules
- `extraction_time_sec`: Time taken for PDF text parsing & Tesseract OCR
- `summary_<model_name>`: Generated 150-word brochure summary per model
- `latency_sec_<model_name>`: LLM generation response latency per model

---

## Current Status & Next Steps
- Tesseract OCR setup and environment variables (`.env`) are fully verified.
- Processing script is configured to capture latency and regex-bounded overviews.
- Running small test batches (2 files) to validate boundary extraction before scaling to all 168 courses.
