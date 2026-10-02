#V1.4 use unproblematic folder 215 min, v1.5 fix capitalization, CSV Progress Loss Risk v1.6 Local X Cloud llm parallel, more noise elimination
import io
import json
import os
import re
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import ollama
from openai import OpenAI
import pandas as pd
from PIL import Image
import pytesseract
from rapidfuzz import fuzz, process
import winsound
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1️⃣ Configuration & Environment Setup
# ----------------------------------------------------------------------
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_DIR = Path(r"D:\InternSem7-test\restart\INDIGO courses")
CATALOG_PATH = Path(r"D:\InternSem7-test\catalog_cards.json")

BATCH_NUM = 3
BATCH_SIZE = 54 #30(34min) +15 (17min) +54+69=168 good for not crash

OUTPUT_CSV = Path(f"course_titles_batch_{BATCH_NUM}.csv")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

openrouter_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# ----------------------------------------------------------------------
# 2️⃣ Model Definitions
# ----------------------------------------------------------------------
MODELS = [
    # Ollama - Local (Run sequentially to protect VRAM)
    {"id": "model_1", "label": "llama3.2_3b", "type": "ollama", "name": "llama3.2:3b"},
    {"id": "model_2", "label": "qwen2.5_7b", "type": "ollama", "name": "qwen2.5:7b"},
    {"id": "model_3", "label": "qwen3.8_27b", "type": "ollama", "name": "qwen3.8:27b"},
    
    # OpenRouter - Cloud (Run concurrently in parallel threads)
    {"id": "model_4", "label": "nemotron_120b_free", "type": "openrouter", "name": "nvidia/nemotron-3-super-120b-a12b:free"},
    {"id": "model_5", "label": "gpt_4o_mini_paid", "type": "openrouter", "name": "openai/gpt-4o-mini"},
]

OLLAMA_MODELS = [m for m in MODELS if m["type"] == "ollama"]
OPENROUTER_MODELS = [m for m in MODELS if m["type"] == "openrouter"]

# ----------------------------------------------------------------------
# 3️⃣ Prompts & Utilities
# ----------------------------------------------------------------------
TITLE_SYSTEM_PROMPT = """You are an expert curriculum design specialist.
Your task is to analyze course document text and generate a compelling, professional Title for the course.

RULES:
1. CAPITALIZATION: Output MUST be in Title Case (e.g., "Advanced Strategic Management for Corporate Leaders").
2. LENGTH: Keep the title between 4 and 15 words. Help recommendation agents decide suitable recommendations by reading the title alone.
3. CONTENT FOCUS: Base the title purely on core skills, learning outcomes, and technical domains in the text. Ignore noise like dates, break times, venues, or HRDF details.
4. STRICT FORMAT: Return ONLY the title string. Do NOT use quotation marks, markdown wrappers, or conversational prefixes (e.g., DO NOT say "Here is the title:").
"""


def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_raw_pdf_text(pdf_path: Path) -> str:
    """Extracts native text and seamlessly falls back to Tesseract OCR for scanned pages."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        native_text = page.get_text("text").strip()

        if len(native_text) < 150:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img).strip()

                page_text = ocr_text if len(ocr_text) > len(native_text) else native_text
            except Exception:
                page_text = native_text
        else:
            page_text = native_text

        if page_text.strip():
            full_text.append(page_text.strip())

    doc.close()
    return "\n\n".join(full_text)


def global_noise_cleaner(text: str) -> str:
    """Strips common marketing headers, schedule noise, contact info, 
    and cuts off trailing boilerplate pages.
    """
    # ------------------------------------------------------------------
    # 1. CUT OFF END-OF-DOCUMENT MARKETING / FOOTER PAGES
    # Cut off text starting from 'WHY CHOOSE US' or 'ABOUT ELITE INDIGO'
    # ------------------------------------------------------------------
    text = re.split(r"(?i)\b(?:WHY\s+CHOOSE\s+US|ABOUT\s+ELITE\s+INDIGO)\b", text)[0]

    # ------------------------------------------------------------------
    # 2. STRIP CONTACT INFO, URLS & BRANDING
    # ------------------------------------------------------------------
    text = re.sub(r"(?i)\bby\s+elite\s+indigo\b", "", text)
    text = re.sub(r"(?i)\belite\s+indigo\s+(sdn\s+bhd|pte\s+ltd)?\b", "", text)
    text = re.sub(r"https?://\S+|www\.\S+|\b\S*eliteindigo\S*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    text = re.sub(r"\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "", text)
    text = re.sub(r"(?i)100%\s+hrdf\s+claimable|hrdcorp\s+claimable|registered\s+hrdcorp\s+training\s+provider", "", text)

    # ------------------------------------------------------------------
    # 3. STRIP SCHEDULE & TIME NOISE
    # Removes 'Lunch', '30 Minutes', '1 Hour', '2 Hours', 'Course Schedule', etc.
    # ------------------------------------------------------------------
    # Standalone Schedule Headers & Timings
    text = re.sub(r"(?i)\b(?:course\s+schedule|day/time|recap\s*&\s*summary\s*of\s*day\s*\d+)\b", "", text)
    text = re.sub(r"(?i)\b\d+\s*(?:full-day|full\s*day)\s*workshop\b", "", text)
    text = re.sub(r"(?i)^\s*(?:lunch|opening\s*&\s*introduction|registration\s*&\s*ice-breaking)\s*$", "", text, flags=re.MULTILINE)
    
    # Standalone duration lines e.g. "1 Hour", "30 Minutes", "3 Hours", "1 Hour 30 Minutes"
    text = re.sub(r"(?i)^\s*(?:\d+\s*(?:hours?|hrs?|minutes?|mins?)\s*)+\s*$", "", text, flags=re.MULTILINE)
    
    # Inline time ranges e.g. "09:00 AM - 05:00 PM" or "(7 Hours)"
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\s*[\u2013\u2014\-]\s*\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*(?:hour|hours|hr|hrs|minute|minutes|min|mins)\s*\)", "", text, flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # 4. CLEAN OCR SYMBOL ARTIFACTS & FORMATTING NOISE
    # ------------------------------------------------------------------
    # Replace weird bullet/OCR characters (e.g. ¢, \f-, ™) with clean text
    text = re.sub(r"[¢\f\f-]", "", text)
    # Remove standalone 1-2 character garbage lines (e.g., 'g', 'aS', 'oo ;')
    text = re.sub(r"^\s*[a-zA-Z0-9;.,\-_]{1,2}\s*$", "", text, flags=re.MULTILINE)

    # ------------------------------------------------------------------
    # 5. DEDUPLICATE LINES & CLEAN BLANK LINES
    # ------------------------------------------------------------------
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Remove exact duplicate consecutive lines (common in slide titles)
    cleaned_lines = []
    for line in lines:
        if not cleaned_lines or line.lower() != cleaned_lines[-1].lower():
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Cap to max characters
    return cleaned[:12000]


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


def clean_title_output(raw_title: str) -> str:
    """Strips quotes, structural headers, and applies clean Title Case without apostrophe artifacts."""
    clean = re.sub(r'^[#*"`\s]+|[#*"`\s]+$', "", raw_title.strip())
    clean = re.sub(r"^(title|generated title):\s*", "", clean, flags=re.IGNORECASE)
    clean = clean.title()
    clean = re.sub(r"('|\b’)([A-Z])\b", lambda m: m.group(1) + m.group(2).lower(), clean)
    return clean

def fix_acronym_casing(text: str) -> str:
    acronyms = ["AI", "PLC", "ISO", "DMAIC", "HRDF", "GRI", "SASB", "TCFD", "CMMS", "EOQ"]
    for acronym in acronyms:
        # Replaces word boundary matches like \bPlc\b or \bAi\b with proper uppercase
        text = re.sub(rf"\b{acronym}\b", acronym, text, flags=re.IGNORECASE)
    return text

# ----------------------------------------------------------------------
# 4️⃣ LLM Title Generation Call
# ----------------------------------------------------------------------
def generate_title_llm(model_config: dict, cleaned_text: str, max_retries: int = 5) -> tuple[str, float]:
    prompt = f"Extract and generate a suitable course title based on the following complete course text:\n\n{cleaned_text}"
    start_time = time.time()

    for attempt in range(1, max_retries + 1):
        try:
            raw_out = ""
            if model_config["type"] == "ollama":
                response = ollama.chat(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_out = response.get("message", {}).get("content", "").strip()

            elif model_config["type"] == "openrouter":
                response = openrouter_client.chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_out = response.choices[0].message.content.strip()

            elapsed_time = round(time.time() - start_time, 2)
            formatted_title = clean_title_output(raw_out)
            return formatted_title, elapsed_time

        except Exception as e:
            sleep_time = 2**attempt
            if attempt < max_retries:
                time.sleep(sleep_time)
                continue

            elapsed_time = round(time.time() - start_time, 2)
            return f"Error: {e}", elapsed_time

# ----------------------------------------------------------------------
# 5️⃣ Main Pipeline Execution
# ----------------------------------------------------------------------
def main():
    print("🚀 Starting Hybrid Parallel Batch Title Generation Pipeline...")

    catalog = load_catalog(CATALOG_PATH)
    catalog_dict = {
        str(item["course_id"]): item
        for item in catalog
        if "course_id" in item
    }

    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    print(f"✅ Loaded {len(pdf_files)} clean PDF files from '{PDF_DIR.name}'.")

    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if "id" in existing_df.columns:
                processed_ids = set(existing_df["id"].astype(str).tolist())
                print(f"📋 Found {len(processed_ids)} previously processed items in '{OUTPUT_CSV.name}'.\n")
        except Exception as e:
            print(f"⚠️ Could not read existing CSV ({e}). Starting clean.")

    all_target_ids = list(catalog_dict.keys())
    start_idx = (BATCH_NUM - 1) * BATCH_SIZE
    end_idx = min(BATCH_NUM * BATCH_SIZE, len(all_target_ids))
    target_ids = all_target_ids[start_idx:end_idx]

    print(f"📋 Loaded {len(all_target_ids)} total catalog items.")
    print(f"⚙ Running Batch {BATCH_NUM}: Items {start_idx} to {end_idx - 1}")

    base_cols = ["id", "pdf_file", "original_title", "extracted_content_sample"]
    title_cols = [f"title_{m['label']}" for m in MODELS]
    latency_cols = [f"latency_sec_{m['label']}" for m in MODELS]
    ordered_columns = base_cols + title_cols + latency_cols

    for idx, target_id in enumerate(target_ids, 1):
        if target_id in processed_ids:
            print(f"⏩ [{idx}/{len(target_ids)}] Skipping ID: {target_id} (Already processed)")
            continue

        matched_item = catalog_dict[target_id]
        original_title = matched_item.get("title", "")

        matched_pdf = find_matching_pdf(target_id, original_title, pdf_files)
        if not matched_pdf:
            print(f"⚠️ [{idx}/{len(target_ids)}] PDF not found for ID: {target_id}")
            continue

        raw_text = extract_raw_pdf_text(matched_pdf)
        cleaned_text = global_noise_cleaner(raw_text)

        check_window = cleaned_text[:4000].lower()
        match_score = fuzz.partial_ratio(original_title.lower(), check_window)
        
        if match_score < 45:
            print(f"\n⚠️ [{idx}/{len(target_ids)}] MISMATCH GUARD TRIGGERED for ID: {target_id}")
            print(f"   Catalog Title: '{original_title}' vs PDF File: '{matched_pdf.name}' (Score: {match_score:.1f})")
            print("   ⏩ Skipping generation to prevent invalid output.\n")
            continue

        print(f"\n[{idx}/{len(target_ids)}] ID: {target_id} | Original Title: {original_title}")

        row_data = {
            "id": target_id,
            "pdf_file": matched_pdf.name,
            "original_title": original_title,
            "extracted_content_sample": cleaned_text[:4000],
        }

        # --- HYBRID PARALLEL MODEL EXECUTION ---
        
        # A. OpenRouter API Calls in Parallel Threads
        def run_openrouter_task(m):
            print(f"  ├─ [Cloud Parallel] Requesting {m['label']}...")
            gen_title, latency = generate_title_llm(m, cleaned_text)
            return f"title_{m['label']}", gen_title, f"latency_sec_{m['label']}", latency

        openrouter_results = {}
        with ThreadPoolExecutor(max_workers=len(OPENROUTER_MODELS)) as executor:
            futures = [executor.submit(run_openrouter_task, m) for m in OPENROUTER_MODELS]
            for future in as_completed(futures):
                t_col, t_val, l_col, l_val = future.result()
                openrouter_results[t_col] = t_val
                openrouter_results[l_col] = l_val

        # B. Ollama Local Calls Sequentially (Single Thread / Protected GPU)
        ollama_results = {}
        for m in OLLAMA_MODELS:
            col_name = f"title_{m['label']}"
            latency_col = f"latency_sec_{m['label']}"
            print(f"  ├─ [Local Sequential] Generating with {m['label']}...")
            gen_title, latency = generate_title_llm(m, cleaned_text)
            ollama_results[col_name] = gen_title
            ollama_results[latency_col] = latency

        # Combine results
        row_data.update(openrouter_results)
        row_data.update(ollama_results)

        # Immediate Save per course item
        df_single = pd.DataFrame([row_data])[ordered_columns]
        file_exists = OUTPUT_CSV.exists()
        
        df_single.to_csv(
            OUTPUT_CSV,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig"
        )

        processed_ids.add(target_id)

    print(f"\n✅ Batch {BATCH_NUM} Complete! Saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
    winsound.MessageBeep(winsound.MB_OK)
