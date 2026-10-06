# Title Generation & Course Intelligence Platform
## Architecture and Improvement Roadmap v2

---

# 1. Project Goal

Current Objective:

Process approximately 168 course PDFs and generate high-quality course titles using local and cloud LLMs.

Future Objective:

Create a reusable content intelligence pipeline that can:

- Extract course information
- Detect existing titles
- Generate improved titles
- Classify courses
- Produce keywords and metadata
- Support recommendation systems
- Support semantic search
- Support future RAG implementations

---

# 2. Current Challenges

Current workflow:

PDF
 ↓
Extraction
 ↓
Cleaning
 ↓
LLM
 ↓
CSV

Problems:

- Extraction and generation tightly coupled
- Existing titles are often ignored
- OCR quality not measured
- Headers and footers add noise
- Document structure is lost
- Difficult to benchmark models
- Difficult to rerun only one stage

---

# 3. Proposed Architecture

Stage 1
PDF Intelligence Extraction

↓

Stage 2
Content Repository

↓

Stage 3
Title Generation

↓

Stage 4
Evaluation & Validation

↓

Stage 5
Recommendation & Search Layer

---

# 4. Stage 1 - PDF Intelligence Extraction

Purpose:

Create a high-quality structured representation of every PDF.

Output:

JSON

Example:

{
    "file": "course123.pdf",
    "pdf_metadata_title": "",
    "cover_page_title": "",
    "overview": "",
    "objectives": "",
    "learning_outcomes": "",
    "keywords": [],
    "ocr_used": false,
    "ocr_quality": "GOOD"
}

---

## 4.1 PDF Metadata Extraction

Extract:

- Title
- Subject
- Author
- Keywords

Source:

PyMuPDF metadata

Store:

pdf_metadata_title

---

## 4.2 Existing Course Title Detection

Priority:

Very High

Many PDFs already contain professionally curated titles.

Detect from:

- Metadata
- Cover page
- Main heading
- Large font text
- First page banner

Store:

cover_page_title

---

## 4.3 Filename Intelligence

Example:

FCI-023-Strategic-Procurement.pdf

Extract:

Strategic Procurement

Store:

filename_title

---

## 4.4 OCR Processing

Use:

- Native text extraction first
- OCR fallback only when needed

Preferred OCR:

- PaddleOCR
- Tesseract

Store:

ocr_used

---

## 4.5 OCR Quality Scoring

Store:

ocr_quality_score

Measurements:

- dictionary ratio
- word confidence
- character quality
- OCR noise ratio

Ratings:

GOOD
MEDIUM
POOR

---

## 4.6 Header and Footer Detection

Automatically remove:

- page numbers
- company branding
- repetitive banners
- website addresses
- contact details

Detect lines repeated across pages.

Avoid vendor-specific rules.

---

## 4.7 Structural Extraction

Preserve sections:

- Overview
- Objectives
- Learning Outcomes
- Course Content
- Modules
- Benefits
- Target Audience

Do not flatten entire document into plain text.

---

## 4.8 Section Weighting

High Value Sections:

1. Title
2. Overview
3. Objectives
4. Learning Outcomes
5. Target Audience

Low Value Sections:

- Registration
- Schedule
- Venue
- Contact Information
- Testimonials

---

## 4.9 Keyword Extraction

Extract:

- technical keywords
- business keywords
- industry keywords

Store:

keywords

Benefits:

- better title generation
- recommendations
- search
- clustering

---

# 5. Stage 2 - Content Repository

Purpose:

Decouple extraction from generation.

Store structured JSON results.

Example:

project/

├── pdf/
├── extracted/
├── generated/
├── evaluation/
└── reports/

Benefits:

- rerun extraction without LLM
- rerun LLM without extraction
- easier debugging
- lower cost
- faster experimentation

---

# 6. Stage 3 - Title Generation

Input:

Extracted JSON

Not PDF files.

---

## 6.1 Title Candidate Sources

Collect:

1. PDF Metadata Title
2. Cover Page Title
3. Filename Title
4. LLM Generated Title

Store all candidates.

---

## 6.2 Title Decision Hierarchy

Priority:

Existing Title
    ↓
Cover Title
    ↓
Metadata Title
    ↓
Filename Title
    ↓
LLM Generated Title

LLM should become a fallback or enhancement layer.

Not the first source.

---

## 6.3 Title Preservation Logic

Modes:

EXACT
MODIFIED
ENHANCED
NEW

Examples:

EXACT

Leadership Essentials

MODIFIED

Leadership Essentials
→ Modern Leadership Essentials

ENHANCED

Leadership Essentials
→ Leadership Essentials for High Performance Teams

NEW

Generated only when no title exists.

---

## 6.4 Prompt Engineering

Recommended Prompt:

Step 1

Identify:

- domain
- core skills
- audience
- learning outcomes

Step 2

Generate:

- professional title
- academic title
- industry title

Return professional title only.

---

## 6.5 Recommended Models

Low

Gemma 3 4B

Moderate

Qwen 3.5 8B or 9B

Advanced

Qwen 3.8 27B

Observation:

Extraction quality contributes more than model size.

---

# 7. Stage 4 - Evaluation Layer

Purpose:

Measure output quality objectively.

---

## 7.1 Confidence Scoring

Store:

confidence_score

Based on:

- keyword overlap
- title similarity
- model agreement
- structural consistency

---

## 7.2 Consensus Scoring

Multiple models:

- Gemma
- Qwen
- Llama
- Mistral
- GPT

Generate:

consensus_score

Ratings:

HIGH
MEDIUM
LOW

---

## 7.3 Course Mismatch Detection

Current:

Fuzzy title comparison

Future:

Combine:

- title similarity
- heading similarity
- keyword similarity

---

## 7.4 Duplicate Detection

Identify:

- duplicate courses
- revised courses
- versioned courses

Store:

course_cluster_id

Applications:

- content cleanup
- curriculum mapping
- recommendations

---

# 8. Stage 5 - Course Intelligence Layer

Future roadmap.

---

## 8.1 Domain Classification

Examples:

AI

Leadership

Manufacturing

Data Science

Power Platform

Quality

HR

Cybersecurity

Store:

domain

subcategory

---

## 8.2 Audience Classification

Store:

- Executive
- Manager
- Supervisor
- Engineer
- Analyst
- General Employee

---

## 8.3 Difficulty Classification

Store:

BEGINNER
INTERMEDIATE
ADVANCED

---

## 8.4 Course Taxonomy

Example:

Quality

├── SPC
├── MSA
├── FMEA
├── APQP
└── IATF

AI

├── AI Fundamentals
├── Prompt Engineering
├── Copilot
├── LLM
└── GenAI

---

## 8.5 Recommendation Readiness

Store:

course_id

title

keywords

domain

subcategory

audience

difficulty

learning_outcomes

cluster_id

embedding

This supports:

- search
- recommendations
- RAG
- learning pathways

---

# 9. Recommended Development Priority

Phase 1

✅ Separate Extraction and Generation

✅ Existing Course Title Detection

✅ Cover Page Title Extraction

✅ Metadata Extraction

---

Phase 2

✅ Header/Footer Detection

✅ OCR Quality Scoring

✅ Structured Section Extraction

✅ Keyword Extraction

---

Phase 3

✅ Confidence Scoring

✅ Consensus Scoring

✅ Duplicate Detection

✅ Domain Classification

---

Phase 4

✅ Taxonomy Builder

✅ Recommendation Engine

✅ Vector Search

✅ RAG Integration

---

# Key Principle

For this project:

Better extraction > Better prompts > Bigger models

A well-structured extraction pipeline using Qwen 9B will often outperform a poor extraction pipeline using Qwen 27B.
