# Context Handover: Course Catalog PDF Processing & Model Comparison Pipeline

## Project Overview & Objectives

* **Context:** Internship at a research center evaluating automated document processing, structured extraction, and marketing summary generation workflows.
* **Task:** Process a batch of **168 course PDFs** stored in nested category subfolders (e.g., `D:\Intern_Sem7\test_pdfs\...`), match them against a master JSON metadata file (`catalog_cards.json`), extract structured attributes (overview, day-by-day outline schedule), generate 150-word marketing brochure summaries using LLMs, compute automated semantic evaluation metrics, and export everything into a consolidated CSV dataset (`research_course_summaries.csv`).
* **Research Goal:** Benchmark performance across local open-source models (via Ollama) versus cloud-hosted free models (via OpenRouter) by recording extraction/OCR latency, inference latency per model, and semantic similarity quality scores.

---

## Technical & Cost Constraints

1. **Budget Cap:** Hard limit of **RM 50 (~$11–$12 USD)**. The pipeline relies primarily on local open-source models and free API tiers to eliminate paid vision/LLM API calls.
2. **Local Environment & Dependencies:**
* **OS:** Windows
* **OCR Engine:** Tesseract OCR `v5.5.3` local binary (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
* **Local LLM Runtime:** Ollama local server managing open-source model inference.
* **Embedding Engine:** `sentence-transformers` running `all-MiniLM-L6-v2` locally for semantic similarity evaluation.
* **Environment Management:** API keys (`OPENROUTER_API_KEY`) and URLs (`OPENROUTER_BASE_URL`) loaded dynamically via `python-dotenv`.



---

## Data Extraction, Preprocessing & Parsing Strategy

### 1. Advanced Hybrid Text/OCR Extraction (`extract_pdf_data`)

* **Native PDF Parsing:** PyMuPDF (`fitz`) attempts raw text extraction page-by-page.
* **Scanned Page Fallback (Scenario A):** Triggers full-page rendering via PIL and Tesseract OCR if native text is ≤ 50 characters.
* **Hybrid Page Parsing (Scenario B):** If native text exists but lacks schedule keywords (`module`, `schedule`, `outline`, etc.), the script extracts embedded images using PyMuPDF image xrefs and OCRs individual image elements directly.
* **PDF Classification:** Classifies each document into `pdf_type` (`"img"` if >50% of pages require full OCR, else `"txt"`).

### 2. Spacing & Noise Cleaning Pipeline (`clean_pdf_noise`)

* **Spaced Character Collapsing (`fix_spaced_text`):** Uses regex (`(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)`) to fix broken single-character spacing (e.g., converting `"O v e r v i e w"` back to `"Overview"`).
* **Corporate Noise Stripping:** Strips web links, emails, HRDF/HRDCorp claimable badges, company marketing footers (*"About Elite Indigo"*, *"Why Choose Us?"*), and schedule timing noise (*"Lunch break"*, *"30 Minutes"*, *"Opening & Recap"*).

### 3. Boundary-Based Overview & Schedule Extraction

* **Overview Slicing (`extract_targeted_overview`):** Bounded from keywords like `Overview`, `Course Overview`, or `Introduction` up to section headers like `Learning Objectives`, `Course Schedule`, or `Target Audience`.
* **Raw Schedule Slicing (`extract_raw_schedule_block`):** Locates start markers like `Module 1`, `Course Schedule`, or `Program Outline` and extracts up to 6,000 characters while stripping timing headers and footer metadata.

### 4. Automated LLM Outline Formatting (`format_outline_with_llm`)

* **Dual-Purpose Role for Model 1:** Before running summary evaluations, **Model 1 (`ollama/qwen2.5:7b`)** acts as a structured syllabus parser.
* **Function:** Transforms noisy, raw OCR schedule text into clean, standardized Markdown containing day assignments (`Day 1`, `Day 2`), module titles, and sub-topic bullet points while ensuring no modules are skipped.

### 5. Catalog Matching & Classification

* **Fuzzy Catalog Matching:** Uses `rapidfuzz` (`token_sort_ratio`) with a score threshold (>60) to match PDF file stems against titles in `catalog_cards.json`.
* **Taxonomy Preservation:** Captures parent directory folder names (e.g., `Artificial Intelligence`) to store as the course category.

---

## Benchmark Models Evaluated

The pipeline benchmarks **3 distinct model endpoints** simultaneously:

| Model ID | Endpoint / Type | Model Name | Primary Task / Role |
| --- | --- | --- | --- |
| **MODEL_1** | Ollama (Local) | `qwen2.5:7b` | **1.** Formats raw OCR into Markdown outlines (`format_outline_with_llm`)<br>

<br>**2.** Generates brochure summary & latency metrics |
| **MODEL_2** | Ollama (Local) | `mistral:latest` | Generates brochure summary & latency metrics |
| **MODEL_3** | OpenRouter (Cloud Free) | `inclusionai/ling-3.0-flash-vl:free` | Generates brochure summary & latency metrics |

*Note: The script includes an automated fallback mechanism (`DISABLED_MODELS`) that disables cloud models gracefully upon hitting API rate limits or quota exhaustion.*

---

## Automated Semantic Similarity Evaluation (`calculate_semantic_similarity`)

* **Ground Truth Grounding:** Builds a composite reference text for each document combining the regex-extracted overview and the LLM-formatted module outline (`rich_reference_text = f"{orig_overview}\n\nSyllabus Outline:\n{outline_summary}"`).
* **Metric Computation:** Uses local embeddings (`all-MiniLM-L6-v2`) to calculate Cosine Similarity (range `0.0` to `1.0`) between each model's generated brochure summary and the document's ground truth content.

---

## Dataset Output Schema (`research_course_summaries.csv`)

| Column Name | Description |
| --- | --- |
| `id` | Course ID matched from `catalog_cards.json` |
| `category` | Subfolder name (e.g., *Artificial Intelligence*) |
| `title` | Matched course title |
| `pdf_type` | Classification of source document (`"txt"` vs `"img"`) |
| `original_overview` | Cleanly extracted overview text (bounded by regex) |
| `outline_summary` | Structured Markdown schedule formatted by `MODEL_1` (`qwen2.5:7b`) |
| `extraction_time_sec` | Total time taken for PyMuPDF parsing, Tesseract OCR, and outline formatting |
| `summary_<model_name>` | Generated 150-word marketing brochure summary per model |
| `latency_sec_<model_name>` | Generation response time in seconds per model |
| `semantic_score_<model_name>` | Cosine similarity score (0.0 to 1.0) evaluating summary against ground truth |

---

## Current Status & Next Steps

* Full end-to-end script (`test_batch_research.py`) implemented with dual local Ollama models (`qwen2.5:7b`, `mistral:latest`), 1 free OpenRouter model, and local semantic evaluation (`SentenceTransformer`).
* Robust hybrid OCR and single-letter spacing cleanup validated on edge-case PDFs.
* File lock fallback included (saves to `research_course_summaries_latest.csv` if main file is open in Excel).
* **Next Step:** Execute the full batch run across all 168 course PDFs and analyze the resulting CSV performance/quality metrics.