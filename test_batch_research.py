import os
import re
import time
import json
import pandas as pd
from pathlib import Path
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image
import io
import ollama
from openai import OpenAI
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from IPython.display import display, Markdown
## v9- fixing hybrid img-txt pdf
# attempting on 11 files

# Load lightweight local embedding model for Semantic Similarity
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

BASE_PDF_FOLDER = Path(r"D:\Intern_Sem7\test_pdfs")
CATALOG_PATH = Path("catalog_cards.json")
OUTPUT_CSV_PATH = Path("research_course_summaries.csv")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Benchmark Models Configuration
MODEL_1 = {"type": "ollama", "name": "qwen2.5:7b"} 
MODEL_2 = {"type": "ollama", "name": "mistral:latest"}
MODEL_3 = {"type": "openrouter", "name": "inclusionai/ling-3.0-flash-vl:free"}

DISABLED_MODELS = {
    "MODEL_1": False,
    "MODEL_2": False,
    "MODEL_3": False,
    "FORMATTER": False
}

# SYSTEM PROMPT: Instructs LLM to read full course details and filter corporate noise
SYSTEM_PROMPT = """You are an expert curriculum marketing specialist.
Your task is to analyze the entire course document and generate a new, engaging 150-word brochure summary.

GUIDELINES:
1. Synthesize core topics across Learning Objectives, Course Outline/Schedule, and Target Audience.
2. Do NOT simply repeat vague overview text; capture the actual syllabus content and practical takeaways.
3. IGNORE ALL NOISE: Omit company contact details, website links, HRDF claimable badges, company reviews, break times, and venue/logistical details.
4. STRICT FORMAT: Plain text only (no markdown bolding, headers, or bullet points).
5. LENGTH: Approximately 150 words."""

openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# Regex Patterns for Boundary Extraction
OVERVIEW_START_PATTERNS = [
    r"program\s+overview", r"course\s+overview", r"overview", r"why\s+matters", 
    r"about\s+this\s+course", r"introduction", r"course\s+description"
]
OVERVIEW_END_PATTERNS = [
    r"learning\s+objective[s]?", r"workshop\s+objective[s]?", r"course\s+objective[s]?",
    r"program\s+outline", r"course\s+schedule", r"pre-requisite[s]?", r"prerequisite[s]?",
    r"learning\s+outcome[s]?", r"expected\s+outcome[s]?", r"target\s+audience"
]

OUTLINE_START_PATTERNS = [
    r"program\s+outline", r"course\s+schedule", r"agenda", 
    r"course\s+outline", r"workshop\s+outline", r"modules", r"course\s+contents"
]
OUTLINE_END_PATTERNS = [
    r"pre-requisite[s]?", r"prerequisite[s]?", r"target\s+audience", 
    r"who\s+should\s+attend", r"learning\s+outcome[s]?", r"trainer\s+profile", r"about\s+elite\s+indigo"
]

def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fix_spaced_text(text: str) -> str:
    """Collapses spaced-out characters (e.g., 'O v e r v i e w' -> 'Overview') 
    while preserving regular single-space gaps between words."""
    # Replaces single-letter space sequences (e.g., 'O v e r v i e w') with merged words
    return re.sub(r'(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)', '', text)

def clean_pdf_noise(raw_text: str) -> str:
    """Pre-processes text to remove spaced characters, URLs, emails, HRDF badges, and corporate footers."""
    # 1. Fix single-character spaced OCR/PDF text first
    text = fix_spaced_text(raw_text)

    # 2. Existing noise cleaning steps
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'(?i)100%\s+hrdf\s+claimable|hrdcorp\s+claimable|registered\s+hrdcorp\s+training\s+provider', '', text)
    text = re.sub(r'(?i)about\s+elite\s+indigo.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)why\s+choose\s+us\?.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    return re.sub(r'\n+', '\n', text).strip()


def extract_targeted_overview(full_text: str) -> str:
    """Extracts text strictly starting from 'Overview' until 'Learning Objectives/Outcome'."""
    start_regex = r"(?i)(" + "|".join(OVERVIEW_START_PATTERNS) + r")"
    end_regex = r"(?i)(" + "|".join(OVERVIEW_END_PATTERNS) + r")"
    
    start_match = re.search(start_regex, full_text)
    if not start_match:
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 80]
        return paragraphs[0] if paragraphs else full_text[:500].replace("\n", " ").strip()
        
    start_idx = start_match.end()
    text_after_start = full_text[start_idx:]
    
    end_match = re.search(end_regex, text_after_start)
    clean_overview = text_after_start[:end_match.start()] if end_match else text_after_start[:1000]
    return re.sub(r"\s+", " ", clean_overview).strip()

def extract_raw_schedule_block(full_text: str) -> str:
    """Extracts complete multi-day schedule blocks and pre-cleans timing/layout noise."""
    # 1. Locate start of schedule content
    start_patterns = [
        r"(?i)module\s+1", r"(?i)course\s+schedule", 
        r"(?i)program\s+outline", r"(?i)course\s+outline"
    ]
    
    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, full_text)
        if match:
            start_idx = match.start()
            break

    # 2. Extract slice up to 6000 characters to cover multi-day PDFs without premature cutoff
    raw = full_text[start_idx:start_idx + 6000]

    # 3. Strip end-of-document marketing footers
    raw = re.split(r"(?i)(about\s+elite\s+indigo|contact\s+us|for\_more\_100%_hrdf)", raw)[0]

    # 4. Remove schedule metadata noise & headers
    noise_patterns = [
        r"(?i)DAY\s*/\s*TIME\s*DESCRIPTION",
        r"(?i)\d+\s*(Hours?|Mins?|Minutes?)\b",
        r"(?i)30\s*Minutes\b",
        r"(?i)Opening\s*&\s*Recap",
        r"(?i)Summary\s*and\s*End\s*of\s*Day\s*\d+",
        r"(?i)Continued\s+on\s+Next\s+Page",
        r"(?i)Lunch(\s*break)?"
    ]
    for pattern in noise_patterns:
        raw = re.sub(pattern, " ", raw)

    # 5. Clean up extra whitespace
    raw = re.sub(r"\n\s*\n", "\n", raw)
    return raw.strip()

def format_outline_with_llm(raw_schedule_text: str) -> str:
    """Formats cleaned outline text into structured Markdown using local Ollama without dropping modules."""
    if not raw_schedule_text or len(raw_schedule_text) < 20:
        return raw_schedule_text

    system_prompt = """You are a precise syllabus parser. Transform raw OCR course schedule text into clean, structured Markdown.

STRICT FORMATTING RULES:
1. COMPLETE MODULE COVERAGE (CRITICAL): 
   - You MUST include EVERY single module present in the input text (from Module 1 up to the last Module). NEVER truncate or skip modules.
2. ACCURATE DAY ASSIGNMENTS:
   - Identify explicit 'Day 1' and 'Day 2' markers in the text.
   - For 2-day courses: Day 1 contains Modules 1 to 3; Day 2 contains Modules 4 to 5.
3. REMOVE NOISE:
   - Omit 'Registration', 'Ice-Breaking', 'Lunch', 'Break', 'Summary and End of Course', 'Opening & Recap', 'Negotiating With the Devil', course titles, page headers.
4. FIX OCR SPACING & TYPOS:
   - Fix merged words (e.g., 'EssentialsFundamentals' -> 'Essentials', 'duringNegotiation' -> 'during Negotiation').
5. STRICT OUTPUT FORMAT ONLY:
   Duration: X Days

   ### Day 1:
   * Module 1: Title
     - Sub-topic description
   * Module 2: Title
     - Sub-topic description


   ### Day 2:
   * Module n: Title
     - Sub-topic description
   * Module n+1: Title
     - Sub-topic description
   * Module n+2: Title
     - Sub-topic description"""

    try:
        response = ollama.chat(
            model=MODEL_1["name"],  # qwen2.5:7b
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Raw Schedule Text:\n{raw_schedule_text}"}
            ],
            options={
                "num_predict": 1500,  # Increases max output generation token limit to prevent cutoffs
                "temperature": 0.0
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Ollama formatting error: {e}")
        return raw_schedule_text

def extract_text_from_page_images(page) -> str:
    """Extracts text from all embedded images inside a PDF page using PyMuPDF and Tesseract OCR."""
    ocr_text = []
    
    # Iterate through all images embedded on the page
    image_list = page.get_images(full=True)
    
    if image_list:
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Open image with Pillow and run Tesseract OCR
                image = Image.open(io.BytesIO(image_bytes))
                extracted = pytesseract.image_to_string(image).strip()
                if extracted:
                    ocr_text.append(extracted)
            except Exception:
                continue

    # Fallback: If no standalone image objects were extracted, render the full page to PNG for OCR
    if not ocr_text:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text.append(pytesseract.image_to_string(img).strip())
        
    return "\n".join(ocr_text).strip()

def extract_pdf_data(pdf_path: Path) -> tuple[str, str, str, str, float]:
    """Extracts PDF text, classifies pdf_type ('txt' vs 'img'), cleans noise, and parses components."""
    start_time = time.time()
    doc = pymupdf.open(pdf_path)
    full_text = []
    ocr_page_count = 0
    total_pages = len(doc)

    schedule_keywords = ["module", "schedule", "outline", "day 1", "agenda", "session", "topic"]

    for page in doc:
        text = page.get_text().strip()
        
        # Scenario A: Scanned Page (Pure Image PDF)
        if len(text) <= 50:
            ocr_page_count += 1
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            full_text.append(pytesseract.image_to_string(img))
            
        # Scenario B: Hybrid Page (Text present, but schedule might be an embedded image)
        else:
            has_schedule_kw = any(kw in text.lower() for kw in schedule_keywords)
            image_list = page.get_images(full=True)
            
            # If text is missing schedule keywords AND page has embedded images, trigger OCR on images
            if not has_schedule_kw and len(image_list) > 0:
                ocr_img_text = extract_text_from_page_images(page)
                if ocr_img_text:
                    text += f"\n\n--- [IMAGE OCR CONTENT] ---\n{ocr_img_text}"
            
            full_text.append(text)

    # Classify pdf_type: 'img' if >50% pages required full-page OCR, else 'txt'
    pdf_type = "img" if (ocr_page_count / max(total_pages, 1)) > 0.5 else "txt"

    raw_combined = "\n\n".join(full_text)
    clean_combined = clean_pdf_noise(raw_combined)

    original_overview = extract_targeted_overview(clean_combined)
    raw_schedule = extract_raw_schedule_block(clean_combined)
    outline_summary = format_outline_with_llm(raw_schedule)

    extraction_time = round(time.time() - start_time, 2)
    return clean_combined, pdf_type, original_overview, outline_summary, extraction_time

def calculate_semantic_similarity(summary_text: str, reference_text: str) -> float:
    """Calculates Semantic Cosine Similarity (0.0 to 1.0) between LLM summary and full syllabus text."""
    if not summary_text or summary_text.startswith("[Error") or summary_text.startswith("[Quota"):
        return None
    
    emb1 = semantic_model.encode(summary_text, convert_to_tensor=True)
    emb2 = semantic_model.encode(reference_text, convert_to_tensor=True)
    cosine_score = util.cos_sim(emb1, emb2).item()
    return round(float(cosine_score), 4)

def match_pdf_to_catalog(pdf_name: str, catalog: list) -> dict:
    clean_name = Path(pdf_name).stem
    titles = [item["title"] for item in catalog]
    best_match, score, index = process.extractOne(clean_name, titles, scorer=fuzz.token_sort_ratio)
    return catalog[index] if score > 60 else None

def generate_llm_summary(model_config: dict, model_key: str, course_title: str, full_cleaned_text: str) -> tuple[str, float]:
    if DISABLED_MODELS.get(model_key, False):
        return None, 0.0

    prompt = f"Course Title: {course_title}\n\nFull Course Document Content:\n{full_cleaned_text[:8000]}"
    start_time = time.time()

    try:
        if model_config["type"] == "ollama":
            response = ollama.chat(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response['message']['content'].strip()

        elif model_config["type"] == "openrouter":
            response = openrouter_client.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content.strip()

        latency = round(time.time() - start_time, 2)
        return result, latency

    except Exception as e:
        err_msg = str(e).lower()
        if any(kw in err_msg for kw in ["insufficient_quota", "rate_limit", "429", "credit", "balance"]):
            DISABLED_MODELS[model_key] = True
            return None, round(time.time() - start_time, 2)

        return f"[Error: {str(e)}]", round(time.time() - start_time, 2)

def main():
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY missing in .env file.")
        return

    catalog = load_catalog(CATALOG_PATH)
    pdf_files = list(BASE_PDF_FOLDER.rglob("*.pdf"))
    rows = []

    print(f"Starting research batch run on {len(pdf_files)} PDFs...\n")

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        category_folder = pdf_file.parent.name
        matched = match_pdf_to_catalog(pdf_file.name, catalog)
        
        if not matched:
            print(f"  └─ ⚠️ Skipped (No catalog match)")
            continue

        try:
            cleaned_text, pdf_type, orig_overview, outline_summary, ocr_time = extract_pdf_data(pdf_file)

            # MODEL GENERATIONS (Reading Full Cleaned Content)
            print("  └─ Running Model 1 (Ollama)...")
            sum1, time1 = generate_llm_summary(MODEL_1, "MODEL_1", matched["title"], cleaned_text)

            print("  └─ Running Model 2 (Ollama)...")
            sum2, time2 = generate_llm_summary(MODEL_2, "MODEL_2", matched["title"], cleaned_text)

            print("  └─ Running Model 3 (OpenRouter Free)...")
            sum3, time3 = generate_llm_summary(MODEL_3, "MODEL_3", matched["title"], cleaned_text)

            # METRICS COMPUTATION
            # Evaluates generated summaries against a rich reference ground truth (Overview + Outline)
            rich_reference_text = f"{orig_overview}\n\nSyllabus Outline:\n{outline_summary}"
            
            print("  └─ Calculating Semantic Similarity Scores...")
            sem1 = calculate_semantic_similarity(sum1, rich_reference_text)
            sem2 = calculate_semantic_similarity(sum2, rich_reference_text)
            sem3 = calculate_semantic_similarity(sum3, rich_reference_text)

            rows.append({
                "id": matched["course_id"],
                "category": category_folder,
                "title": matched["title"],
                "pdf_type": pdf_type,
                "original_overview": orig_overview,
                "outline_summary": outline_summary,
                "extraction_time_sec": ocr_time,
                
                # Model 1 (Ollama)
                f"summary_{MODEL_1['name'].replace(':', '_').replace('/', '_')}": sum1,
                f"latency_sec_{MODEL_1['name'].replace(':', '_').replace('/', '_')}": time1,
                f"semantic_score_{MODEL_1['name'].replace(':', '_').replace('/', '_')}": sem1,

                # Model 2 (Ollama)
                f"summary_{MODEL_2['name'].replace(':', '_').replace('/', '_')}": sum2,
                f"latency_sec_{MODEL_2['name'].replace(':', '_').replace('/', '_')}": time2,
                f"semantic_score_{MODEL_2['name'].replace(':', '_').replace('/', '_')}": sem2,

                # Model 3 (OpenRouter free)
                f"summary_{MODEL_3['name'].replace(':', '_').replace('/', '_')}": sum3,
                f"latency_sec_{MODEL_3['name'].replace(':', '_').replace('/', '_')}": time3,
                f"semantic_score_{MODEL_3['name'].replace(':', '_').replace('/', '_')}": sem3,
            })

        except Exception as e:
            print(f"  └─ ❌ Error on file {pdf_file.name}: {e}")

    df = pd.DataFrame(rows)
    
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\nResearch test complete! Output saved to '{OUTPUT_CSV_PATH}'.")
    except PermissionError:
        fallback_path = Path("research_course_summaries_latest.csv")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"\n⚠️ '{OUTPUT_CSV_PATH}' was locked. Output saved to '{fallback_path}'.")

if __name__ == "__main__":
    main()



