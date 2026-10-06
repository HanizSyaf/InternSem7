# Title Generation Pipeline - Planned Improvements Roadmap

## Project Context
Current system processes 168 PDF course documents and generates suggested course titles using multiple LLMs (Ollama + OpenRouter).

Current strengths:
- OCR fallback using Tesseract
- PDF text extraction via PyMuPDF
- Parallel/cloud model execution
- Noise cleaning
- Fuzzy matching protection
- Batch processing
- Resume capability through CSV output

---

# High-Priority Improvements

## 1. Existing PDF Title Detection

### Current Gap
The system generates titles only from extracted text.

It does NOT currently check:
- PDF metadata title
- Cover-page title
- Main heading
- Course title already printed inside the PDF

### Recommended Solution

#### Source 1: PDF Metadata

Use:

```python
meta = doc.metadata
pdf_title = meta.get("title", "")
```

Store:

```text
pdf_metadata_title
```

#### Source 2: Cover Page Heading

Replace:

```python
page.get_text("text")
```

with:

```python
page.get_text("dict")
```

Extract:

- largest font size
- top-most heading
- first page title blocks

Store:

```text
cover_page_title
```

#### Source 3: Filename Analysis

Example:

```text
FCI-023-Strategic-Procurement.pdf
```

Generate:

```text
Strategic Procurement
```

Store:

```text
filename_title
```

### New Title Selection Hierarchy

```text
PDF Metadata
      ↓
Cover Page Title
      ↓
Detected Main Heading
      ↓
Filename Candidate
      ↓
LLM Generated Title
```

---

## 2. Structured Content Extraction

### Current Situation

Current extraction:

```python
page.get_text("text")
```

removes document structure.

### Improvement

Preserve:

- Course Overview
- Objectives
- Learning Outcomes
- Target Audience
- Module Contents
- Benefits
- Conclusion

Build structured prompts.

Example:

```text
Overview:
...

Objectives:
...

Learning Outcomes:
...
```

This usually improves title generation significantly.

---

## 3. Repeated Header/Footer Removal

### Current Issue

Current cleaner contains Indigo-specific rules.

Example:

```text
WHY CHOOSE US
ABOUT ELITE INDIGO
```

### Improvement

Automatically detect lines appearing on:

```text
50% to 80% of pages
```

Remove:

- page numbers
- company branding
- contact details
- website URLs
- repeated training advertisements

Result:

Cleaner content and better keywords.

---

## 4. OCR Quality Scoring

### Current Situation

OCR activates when:

```python
len(native_text) < 150
```

### Additional Detection

Store:

```text
ocr_used
ocr_quality_score
```

Example checks:

- dictionary word ratio
- alphabetic character ratio
- OCR confidence metrics

Flag weak OCR automatically.

---

## 5. Section-Based Content Selection

### Current Situation

Entire content is truncated:

```python
cleaned_text[:12000]
```

### Improvement

Prioritize:

```text
Overview
Objectives
Learning Outcomes
Modules
Target Audience
```

Ignore:

```text
Schedules
Lunch breaks
Registration details
Contact information
```

Smaller context often produces better titles.

---

## 6. Better Prompt Engineering

### Current Prompt

Generate a suitable course title.

### Recommended Prompt

```text
Step 1:
Identify:
- subject domain
- core skills
- target audience
- learning outcomes

Step 2:
Generate:
- professional title
- academic title
- industry title

Return only the professional title.
```

Qwen models benefit significantly from explicit reasoning.

---

# Model Recommendations

Low:
- Gemma 3 4B

Moderate:
- Qwen 3.5 8B or 9B

Advanced:
- Qwen 3.8 27B

Observation:
Most quality gains will come from extraction quality rather than moving to larger models.

---

# Additional CSV Columns

Recommended additions:

```text
pdf_metadata_title
cover_page_title
filename_title
extracted_keywords
ocr_used
ocr_quality_score
content_length
confidence_score
```

---

# Additional Enhancements Identified From Current Script

## Replace Vendor-Specific Cleaning

Current cleaner is tightly coupled to Elite Indigo content.

Create:

```python
def generic_noise_cleaner():
```

Use reusable patterns suitable for future datasets.

---

## Detect Course Catalog Mismatch Smarter

Current threshold:

```python
fuzz.partial_ratio() < 45
```

Improvement:

Combine:

- title similarity
- keyword similarity
- detected heading similarity

for better confidence.

---

## Add Keyword Extraction Layer

Before title generation:

```text
PDF
 ↓
Keywords
 ↓
Title Generation
```

Store top keywords.

Useful for:

- recommendation systems
- search
- course clustering
- quality validation

---

## Add Confidence Scoring

Store:

```text
confidence_score
```

Based on:

- title agreement between models
- keyword overlap
- detected title similarity

Useful when selecting the best generated title.

---

## Future Architecture

```text
PDF
 ├─ Metadata Extraction
 ├─ File Name Analysis
 ├─ Cover Title Detection
 ├─ OCR Processing
 ├─ Repeated Header Removal
 ├─ Structure Extraction
 ├─ Keyword Extraction
 ├─ LLM Title Generation
 └─ Confidence Scoring

Output
 ├─ Original Title
 ├─ Metadata Title
 ├─ Cover Title
 ├─ Filename Title
 ├─ Generated Titles
 ├─ Keywords
 ├─ OCR Score
 └─ Confidence Score
```


# Additional Findings from Generated Results Review

---

## 7. Existing Title Preservation

### Observation

Many PDFs already contain high-quality, publish-ready titles.

Examples:

Original:
Leading By Listening

Generated:
Leading By Listening: Practical Communication Skills For Effective Leadership

Original:
Microsoft 365 Copilot for Real Work

Generated:
Microsoft 365 Copilot For Real Workplace Productivity And Prompt Mastery

Original:
Natural Language Processing: From Fundamentals to Real-World Applications

Generated:
Natural Language Processing: From Fundamentals To Real-World Applications

In some cases the original title is equal or better than the generated title.

---

### Improvement

Implement title classification before title generation.

Decision flow:

PDF
 ├─ Metadata Title
 ├─ Cover Page Title
 ├─ Existing Course Title
 └─ LLM

Decision:

IF confidence(existing_title) > threshold
    retain existing title
ELSE
    generate alternative title

---

### Benefits

- Better catalogue consistency
- Less model cost
- Less title drift
- Easier benchmarking

---

## 8. Title Drift Detection

### Observation

Models frequently expand titles:

Original:
Workplace Wellness

Generated:
Mastering Workplace Wellness: Strategies For Burnout Prevention And Resilience

Problem:

Generated title may no longer match:

- course catalogue
- brochure
- registration records
- LMS

---

### Improvement

Generate classification:

title_preservation_mode

Values:

EXACT
MODIFIED
ENHANCED
NEW

Examples:

EXACT:
Microsoft 365 Copilot for Real Work

ENHANCED:
Leading By Listening
→ Leading By Listening: Practical Leadership Communication

NEW:
when no usable title exists

---

## 9. Duplicate Course Detection

### Observation

Several files appear to be:

- revisions
- variants
- duplicate course versions

Examples:

ISO 9001 Awareness
ISO 9001:2015 Awareness Training

Growth Mindset
Growth Mindset 2

Tufting Adventure
Tufting Adventure (Hotel)

---

### Improvement

Add duplicate detection:

fuzzy_title_similarity

course_cluster_id

Potential use:

- recommendation engines
- content deduplication
- LMS migration
- curriculum mapping

---

## 10. Domain Classification Layer

### Observation

Generated titles alone are insufficient.

Future recommendation systems need:

- domain
- category
- audience

---

### Example

Title:
AI Enhanced Root Cause Analysis for Manufacturing Excellence

Domain:
Manufacturing

SubDomain:
Quality Engineering

Audience:
Engineers

Difficulty:
Intermediate

Keywords:
RCA
Fishbone
5 Whys
AI

---

### Recommended Output Columns

domain
subcategory
audience
difficulty
keywords

---

## 11. Course Taxonomy Builder

### Observation

The dataset naturally clusters into groups:

AI
Leadership
Communication
Manufacturing
Quality
Data Science
Power Platform
Maintenance
Food Safety
ISO
Team Building

---

### Future Enhancement

Create automatic taxonomy generation.

Example:

AI
 ├─ AI Fundamentals
 ├─ Prompt Engineering
 ├─ LLM
 ├─ Copilot
 └─ Generative AI

Quality
 ├─ SPC
 ├─ MSA
 ├─ FMEA
 ├─ APQP
 └─ IATF

Benefits:

- recommendation systems
- search
- analytics
- course discovery

---

## 12. Multi-Model Consensus Scoring

### Observation

Current output stores:

- Llama
- Qwen
- Qwen 27B
- Nemotron
- GPT-4o Mini

This is valuable data.

---

### Improvement

Create:

consensus_score

Example:

If 4 of 5 models generate similar title:

Consensus:
HIGH

If every model generates different title:

Consensus:
LOW

---

### Formula

title_similarity_average

keyword_overlap

semantic_similarity

---

### Output

consensus_score
title_confidence

---

## 13. OCR Failure Detection

### Observation

Several PDFs contain OCR artifacts:

Examples:

G invico
Nn |f
Random symbols
Broken spacing

---

### Improvement

Measure:

ocr_noise_score

Indicators:

- abnormal character ratio
- dictionary word ratio
- repeated OCR artifacts

---

### Output

ocr_quality

GOOD
MEDIUM
POOR

---

## 14. Section Ranking Engine

Current extraction sends huge blocks of text.

Not all sections contribute equally.

---

### High-value sections

1. Cover title
2. Executive summary
3. Program overview
4. Learning objectives
5. Target audience

---

### Low-value sections

- trainer profile
- testimonials
- registration
- schedule
- venue
- contact information

---

### Improvement

Weighted section extraction

Instead of:

12000 character dump

Use:

Title = 35%
Overview = 30%
Objectives = 20%
Audience = 10%
Keywords = 5%

This improves title quality significantly.

---

## 15. Recommendation-System Readiness

Future-proof output structure.

Add:

course_id
title
keywords
domain
subcategory
audience
difficulty
learning_outcomes
embedding
cluster_id

This makes future:

- semantic search
- RAG
- recommendation engines
- personalized learning paths

much easier to build.

---

# Improvement Priority Ranking

1. Detect Existing Course Title and cover-page title extraction
2. Structured section extraction
3. Automatic header/footer removal
4. OCR quality scoring
5. Metadata title extraction
6. Keyword extraction layer
7. Confidence scoring
8. Larger LLM models

Expected benefit:

The first four improvements will likely contribute more title quality improvement than upgrading from a 9B model to a 27B model.
