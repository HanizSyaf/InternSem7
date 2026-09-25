import io
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

import fitz  # PyMuPDF
import ollama
from openai import OpenAI
import pandas as pd
from PIL import Image
import pytesseract
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
PDF_DIR = Path(r"D:\Intern_Sem7\INDIGO courses")
CATALOG_PATH = Path(r"D:\Intern_Sem7\catalog_cards.json")
STAGE2_CSV = Path("course_summaries_stage2.csv")
FINAL_CSV = Path("course_summaries_stage3_final.csv")

TARGET_IDS = [
    "c-8b999366c1",
    "c-a6d4b241a8",
    "c-55de08e6a5",
    "c-493724d2f6",
    "c-b2cbde9198",
]

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# OpenRouter Configuration (for external benchmark models)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

openrouter_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# Models for Stage 2 Summarization
OLLAMA_MODEL_NAME = "llama3.2:3b"  # Used for Stage 1 Extraction & Stage 2 Local Model

MODEL_1 = {"type": "ollama", "name": OLLAMA_MODEL_NAME}
MODEL_2 = {"type": "openrouter", "name": "nvidia/nemotron-3-super-120b-a12b:free"}
MODEL_3 = {"type": "openrouter", "name": "deepseek/deepseek-chat"}

DISABLED_MODELS = {"MODEL_1": False, "MODEL_2": False, "MODEL_3": False}

# ==========================================
# PROMPTS
# ==========================================
EXTRACTION_SYSTEM_PROMPT = """You are an expert curriculum parser. 
Extract information from raw course text into strict JSON format.

RULES FOR EXTRACTION:
1. OVERVIEW:
   - Extract core high-level course description/summary.
   - FALLBACK RULE: If no dedicated course overview section exists, look for 'Why [Topic/Workshop] Matters', 'Why Attend', or 'Why This Course Matters' sections.
   - IF FALLBACK IS USED: Line 1 = 'Why [Topic/Workshop] Matters', Line 2+ = supporting text.
   - Strip contact info, websites, HRDF details, footers, and noise.
   - IF NEITHER OVERVIEW NOR 'WHY IT MATTERS' CONTENT EXISTS, RETURN null.

2. OUTLINE CLEANING & NOISE STRIPPING:
   - STRIP TIME MARKERS: Remove all timestamps and durations (e.g., '(1 Hour)', '(30 Minutes)', '9.00 am - 10.00 am', '3 Hours').
   - STRIP NON-COURSE ACTIVITIES: Completely omit non-learning items such as:
     * Lunch breaks, tea/coffee breaks, rest breaks
     * Opening activities, welcome, registration, logistics
     * Day wrap-ups, end of day summaries, Q&A closing
   - Format sub-topics in Format A or C as individual array elements starting with '  - '.

3. OUTLINE CLASSIFICATION:
   - Determine training DURATION: e.g., "1", "2", or "Not Specified".
   - Classify OUTLINE FORMAT TYPE as exactly one of:
       * "A": Module-based WITH sub-descriptions/bullets.
       * "B": Module-based WITHOUT sub-descriptions/bullets.
       * "C": Session-based (MUST contain explicit headers like 'Morning Session') WITH sub-descriptions.
       * "D": Session-based (MUST contain explicit headers like 'Morning Session') WITHOUT sub-descriptions.
       * "E": Freeform activity list or agenda flow (none of the above).

STRICT JSON OUTPUT FORMAT (No markdown wrappers):
{
  "overview": "Clean overview string or null",
  "duration": "e.g., '2' or 'Not Specified'",
  "format_type": "A | B | C | D | E",
  "outline_items": [
    "Item heading or bullet matching the rules above"
  ]
}"""


SUMMARY_SYSTEM_PROMPT = """You are an expert curriculum marketing specialist.
Your task is to analyze the course document content and generate an engaging, cohesive ~150-word brochure summary.

GUIDELINES:
1. Synthesize core topics, key modules, target audience, and practical takeaways into smooth paragraphs.
2. Capture actual syllabus content and practical skills learned.
3. IGNORE ALL NOISE: Omit contact details, websites, HRDF badges, break times, and clock schedules.
4. STRICT FORMAT: Plain text only (no markdown bolding, headers, or bullet points).
5. LENGTH: Approximately 150 words."""


# ==========================================
# UTILITY & EXTRACTION FUNCTIONS
# ==========================================
def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_catalog_category(item: dict) -> str:
    """Extracts category tags or fallback category fields from catalog cards."""
    if "category_tags" in item and isinstance(item["category_tags"], list):
        return ", ".join(item["category_tags"])
    
    for key in ["category", "course_category", "group", "domain", "topic"]:
        if key in item and item[key]:
            return str(item[key])
            
    return "Uncategorized"


def global_noise_cleaner(text: str) -> str:
    # URL / Email / HRDF Noise
    text = re.sub(
        r"https?://\S+|www\.\S+|\b\S*eliteindigo\S*\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
    )
    text = re.sub(
        r"(?i)100%\s+hrdf\s+claimable|hrdcorp\s+claimable|registered\s+hrdcorp\s+training\s+provider",
        "",
        text,
    )

    # Time timestamps (e.g., 9.00 am — 10.15 am, 9:00AM - 12:00PM)
    text = re.sub(
        r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\s*[\u2013\u2014\-]\s*\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Hour/minute duration noise (e.g., (1 Hour), (30 Minutes), 3 Hours)
    text = re.sub(
        r"\(\s*\d+\s*(?:hour\vert{}hours\vert{}hr\vert{}hrs\vert{}minute\vert{}minutes\vert{}min\vert{}mins)\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def strip_time_and_operational_noise(text: str) -> str:
    text = re.sub(r"\b\d{1,2}[\.:]\d{2}\s*(?:am|pm)?\s*(?:-|to|–)\s*\d{1,2}[\.:]\d{2}\s*(?:am|pm)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\b(tea break|lunch break|networking lunch|registration|recap & q&a)\b", "", text)
    return global_noise_cleaner(text)


def extract_raw_pdf_text_with_type(pdf_path: Path) -> tuple[str, str]:
    """Extracts PDF text while tracking if PDF is 'txt', 'img', or 'mix'."""
    doc = fitz.open(pdf_path)
    full_text = []
    has_txt = False
    has_img = False

    for page in doc:
        text = page.get_text("text").strip()
        if len(text) >= 50:
            has_txt = True
            full_text.append(text)
        else:
            has_img = True
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            full_text.append(ocr_text)

    if has_txt and has_img:
        pdf_type = "mix"
    elif has_img:
        pdf_type = "img"
    else:
        pdf_type = "txt"

    return "\n\n".join(full_text), pdf_type


def find_matching_pdf(target_id: str, target_title: str, pdf_files: list) -> Path | None:
    for pdf in pdf_files:
        if target_id in pdf.stem or target_id in pdf.name:
            return pdf

    pdf_stems = [pdf.stem for pdf in pdf_files]
    best_match, score, idx = process.extractOne(
        target_title, pdf_stems, scorer=fuzz.token_sort_ratio
    )

    if score > 50:
        return pdf_files[idx]

    return None


def validate_overview(overview: str | None) -> str | None:
    """Guardrail to reject pure list/agenda leaks in overview field."""
    if not overview or not isinstance(overview, str):
        return None
    
    clean_ov = overview.strip()
    if len(clean_ov) < 35:
        return None

    # Reject ONLY if text directly begins with schedule/module headers
    lines = [line.strip().lower() for line in clean_ov.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    
    if re.match(r"^(module\s+\d+|day\s+\d+|session\s+\d+|agenda|schedule)", first_line):
        return None

    return clean_ov


def sanitize_and_parse_json(raw_text: str) -> dict:
    clean_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"^```\s*", "", clean_text)
    clean_text = re.sub(r"\s*```$", "", clean_text).strip()
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        clean_text = clean_text[start_idx : end_idx + 1]
    return json.loads(clean_text)


# ==========================================
# STAGE 1: LOCAL OLLAMA EXTRACTION
# ==========================================
def run_stage1_extraction(cleaned_text: str, course_title: str, max_retries: int = 4) -> dict:
    prompt = f"Course Title: {course_title}\n\nDocument Text:\n{cleaned_text[:7000]}"
    
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
            )
            raw_out = response.get("message", {}).get("content", "").strip()
            parsed_json = sanitize_and_parse_json(raw_out)
            parsed_json["overview"] = validate_overview(parsed_json.get("overview"))
            return parsed_json
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {
                "overview": None,
                "duration": "Error",
                "format_type": "E",
                "outline_items": [f"Parsing Error: {e}"],
            }


# ==========================================
# STAGE 2: MULTI-MODEL SUMMARIZATION
# ==========================================
def generate_llm_summary(
    model_config: dict,
    model_key: str,
    course_title: str,
    stage1_data: dict,
    full_cleaned_text: str,
    max_retries: int = 6,
) -> tuple[str | None, float]:
    if DISABLED_MODELS.get(model_key, False):
        return None, 0.0

    prompt = f"""Course Title: {course_title}
Duration: {stage1_data.get('duration', 'Not Specified')} Days

--- STAGE 1 STRUCTURED CONTEXT ---
Overview: {stage1_data.get('overview') or 'N/A'}
Outline Format: {stage1_data.get('format_type')}
Extracted Outline Items:
{json.dumps(stage1_data.get('outline_items', []), indent=2)}

--- FULL COURSE DOCUMENT CONTENT ---
{full_cleaned_text}
"""
    start_time = time.time()

    for attempt in range(1, max_retries + 1):
        try:
            if model_config["type"] == "ollama":
                response = ollama.chat(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                summary = response.get("message", {}).get("content", "").strip()

            elif model_config["type"] == "openrouter":
                response = openrouter_client.chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                summary = response.choices[0].message.content.strip()

            return summary, round(time.time() - start_time, 2)

        except Exception as e:
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ["insufficient_quota", "credit", "402"]):
                DISABLED_MODELS[model_key] = True
                break
            if attempt < max_retries:
                time.sleep(2)

    return None, round(time.time() - start_time, 2)


# ==========================================
# STAGE 3: EVALUATION METRICS
# ==========================================
def run_stage3_evaluation(df: pd.DataFrame, semantic_model: SentenceTransformer) -> pd.DataFrame:
    summary_cols = [col for col in df.columns if col.startswith("summary_")]

    for idx, row in df.iterrows():
        # Construct Reference Text
        overview = str(row["overview"]) if pd.notna(row["overview"]) else ""
        outline_raw = row["outline_items"]
        outline_str = ""
        if pd.notna(outline_raw):
            try:
                items = json.loads(outline_raw)
                outline_str = "\n".join(items) if isinstance(items, list) else str(items)
            except Exception:
                outline_str = str(outline_raw)

        reference_text = f"Course Overview:\n{overview}\n\nCourse Outline:\n{outline_str}".strip()
        emb_ref = semantic_model.encode(reference_text, convert_to_tensor=True)

        for sum_col in summary_cols:
            model_tag = sum_col.replace("summary_", "")
            summary_val = row[sum_col]

            if pd.isna(summary_val) or not str(summary_val).strip() or summary_val == "None":
                sem_score, w_count = 0.0, 0
            else:
                emb_sum = semantic_model.encode(str(summary_val), convert_to_tensor=True)
                sem_score = round(util.cos_sim(emb_sum, emb_ref).item(), 4)
                w_count = len(str(summary_val).split())

            df.at[idx, f"semantic_score_{model_tag}"] = sem_score
            df.at[idx, f"word_count_{model_tag}"] = w_count

    return df


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
def main():
    print("🚀 Starting Unified Course Pipeline...")
    catalog = load_catalog(CATALOG_PATH)
    catalog_dict = {item["course_id"]: item for item in catalog}
    pdf_files = list(PDF_DIR.rglob("*.pdf"))

    rows = []
    
    # --------------------------------------
    # STAGES 1 & 2
    # --------------------------------------
    for idx, target_id in enumerate(TARGET_IDS, 1):
        matched_item = catalog_dict.get(target_id)
        if not matched_item:
            continue

        target_title = matched_item["title"]
        matched_pdf = find_matching_pdf(target_id, target_title, pdf_files)
        if not matched_pdf:
            continue

        print(f"\n[{idx}/{len(TARGET_IDS)}] Processing ID: {target_id} | Title: {target_title}")

        # 1. Parsing & Noise Removal
        raw_text, pdf_type = extract_raw_pdf_text_with_type(matched_pdf)
        cleaned_text = global_noise_cleaner(raw_text)

        # 2. Stage 1 Extraction (Local Ollama)
        print("  ├─ Running Stage 1 Extraction (Ollama)...")
        stage1_data = run_stage1_extraction(cleaned_text, target_title)

        # 3. Stage 2 Summarization
        full_text_no_time = strip_time_and_operational_noise(cleaned_text)
        
        print("  ├─ Summarizing with Model 1 (Ollama)...")
        sum1, t1 = generate_llm_summary(MODEL_1, "MODEL_1", target_title, stage1_data, full_text_no_time)
        
        print("  ├─ Summarizing with Model 2 (OpenRouter)...")
        sum2, t2 = generate_llm_summary(MODEL_2, "MODEL_2", target_title, stage1_data, full_text_no_time)
        
        print("  └─ Summarizing with Model 3 (OpenRouter)...")
        sum3, t3 = generate_llm_summary(MODEL_3, "MODEL_3", target_title, stage1_data, full_text_no_time)

        m1_tag = MODEL_1["name"].replace(":", "_").replace("/", "_")
        m2_tag = MODEL_2["name"].replace(":", "_").replace("/", "_")
        m3_tag = MODEL_3["name"].replace(":", "_").replace("/", "_")

        rows.append({
            "id": target_id,
            "title": target_title,
            "pdf_file": matched_pdf.name,
            "overview": stage1_data.get("overview"),
            "duration": stage1_data.get("duration"),
            "format_type": stage1_data.get("format_type"),
            "outline_items": json.dumps(stage1_data.get("outline_items", [])),
            f"summary_{m1_tag}": sum1,
            f"summary_{m2_tag}": sum2,
            f"summary_{m3_tag}": sum3,
            f"latency_sec_{m1_tag}": t1,
            f"latency_sec_{m2_tag}": t2,
            f"latency_sec_{m3_tag}": t3,
        })

    df = pd.DataFrame(rows)
    df.to_csv(STAGE2_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Stage 1 & 2 Complete. Saved to '{STAGE2_CSV}'.")

    # --------------------------------------
    # STAGE 3 EVALUATION
    # --------------------------------------
    print("\n🔍 Loading SentenceTransformer ('all-MiniLM-L6-v2') for Evaluation...")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("📊 Evaluating Semantic Scores and Word Counts...")
    df_evaluated = run_stage3_evaluation(df, semantic_model)

    df_evaluated.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"🎉 Pipeline Complete! Final outputs saved to '{FINAL_CSV}'.")


if __name__ == "__main__":
    main()