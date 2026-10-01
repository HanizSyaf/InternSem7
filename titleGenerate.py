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
import winsound

# ----------------------------------------------------------------------
# 1️⃣ Configuration & Environment Setup
# ----------------------------------------------------------------------
load_dotenv()

PDF_DIR = Path(r"D:\Intern_Sem7\restart\INDIGO courses")
CATALOG_PATH = Path(r"D:\Intern_Sem7\catalog_cards.json")

BATCH_NUM = 1
BATCH_SIZE = 5

OUTPUT_CSV = Path(f"course_titles_batch_{BATCH_NUM}.csv")

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

openrouter_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# ----------------------------------------------------------------------
# 2️⃣ Model Definitions (4 Selected Models)
# ----------------------------------------------------------------------
MODELS = [
    # 1️⃣ Ollama - Moderate
    {
        "id": "model_1",
        "label": "llama3_2_3b",
        "type": "ollama",
        "name": "llama3.2:3b",
    },
    # 2️⃣ Ollama - Advanced
    {
        "id": "model_2",
        "label": "qwen3_5_latest",
        "type": "ollama",
        "name": "qwen3.5:latest",
    },
    # 3️⃣ OpenRouter - Moderate (Free)
    {
        "id": "model_3",
        "label": "nemotron_120b_free",
        "type": "openrouter",
        "name": "nvidia/nemotron-3-super-120b-a12b:free",
    },
    # 4️⃣ OpenRouter - Advanced (Paid Benchmark)
    {
        "id": "model_4",
        "label": "gpt_4o_mini_paid",
        "type": "openrouter",
        "name": "openai/gpt-4o-mini",
    },
]

# ----------------------------------------------------------------------
# 3️⃣ Extraction & Cleaning Utilities
# ----------------------------------------------------------------------
TITLE_SYSTEM_PROMPT = """You are an expert curriculum design specialist.
Your task is to analyze course document text and generate a compelling, professional Title for the course.

RULES:
1. CAPITALIZATION: Output MUST be in Title Case (e.g., "Advanced Strategic Management for Corporate Leaders").
2. LENGTH: Keep the title between 4 and 15 words. something that can help hermes agent decide suitable course recommendation by reading title alone.
3. CONTENT FOCUS: Base the title purely on core skills, learning outcomes, and technical domains in the text. Ignore noise like dates, break times, venues, or HRDF details.
4. STRICT FORMAT: Return ONLY the title string. Do NOT use quotation marks, markdown wrappers, or conversational prefixes (e.g., DO NOT say "Here is the title:").
"""


def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def global_noise_cleaner(text: str) -> str:
    # Remove brand header/footer noise
    text = re.sub(r"(?i)\bby\s+elite\s+indigo\b", "", text)
    text = re.sub(r"(?i)\belite\s+indigo\s+(sdn\s+bhd|pte\s+ltd)?\b", "", text)
    
    # URL / Email / HRDF Noise
    text = re.sub(r"https?://\S+|www\.\S+|\b\S*eliteindigo\S*\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    text = re.sub(r"(?i)100%\s+hrdf\s+claimable|hrdcorp\s+claimable|registered\s+hrdcorp\s+training\s+provider", "", text)
    
    # Time / Schedule noise
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\s*[\u2013\u2014\-]\s*\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*(?:hour|hours|hr|hrs|minute|minutes|min|mins)\s*\)", "", text, flags=re.IGNORECASE)
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_raw_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text("text").strip()
        
        # If page has minimal native text, run OCR to capture diagram/scanned text
        if len(page_text) < 300:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img).strip()
                
                # Append OCR text if it captured more content than native PyMuPDF
                if len(ocr_text) > len(page_text):
                    page_text = f"{page_text}\n{ocr_text}"
            except Exception as e:
                pass  # Fall back to native page_text if OCR fails
                
        if page_text:
            full_text.append(page_text)

    return "\n\n".join(full_text)


def find_matching_pdf(
    target_id: str, target_title: str, pdf_files: list
) -> Path | None:
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
    """Strips quotes, structural headers, and ensures Title Case."""
    clean = re.sub(r'^[#*"`\s]+|[#*"`\s]+$', "", raw_title.strip())
    clean = re.sub(r"^(title|generated title):\s*", "", clean, flags=re.IGNORECASE)
    return clean.title()


# ----------------------------------------------------------------------
# 4️⃣ LLM Title Generation Call
# ----------------------------------------------------------------------
def generate_title_llm(
    model_config: dict, cleaned_text: str, max_retries: int = 5
) -> tuple[str, float]:
    # Send the COMPLETE cleaned text of the entire document
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
    print("🚀 Starting Batch Title Generation Pipeline...")
    catalog = load_catalog(CATALOG_PATH)
    catalog_dict = {item["course_id"]: item for item in catalog if "course_id" in item}
    pdf_files = list(PDF_DIR.rglob("*.pdf"))

    all_target_ids = list(catalog_dict.keys())
    start_idx = (BATCH_NUM - 1) * BATCH_SIZE
    end_idx = min(BATCH_NUM * BATCH_SIZE, len(all_target_ids))
    target_ids = all_target_ids[start_idx:end_idx]

    print(f"📋 Loaded {len(all_target_ids)} total catalog items.")
    print(f"⚙️ Running Batch {BATCH_NUM}: Items {start_idx} to {end_idx - 1}")

    rows = []

    for idx, target_id in enumerate(target_ids, 1):
        matched_item = catalog_dict[target_id]
        original_title = matched_item.get("title", "")

        matched_pdf = find_matching_pdf(target_id, original_title, pdf_files)
        if not matched_pdf:
            print(f"⚠️ [{idx}/{len(target_ids)}] PDF not found for ID: {target_id}")
            continue

        print(f"\n[{idx}/{len(target_ids)}] ID: {target_id} | Original Title: {original_title}")

        raw_text = extract_raw_pdf_text(matched_pdf)
        cleaned_text = global_noise_cleaner(raw_text)

        row_data = {
            "id": target_id,
            "pdf_file": matched_pdf.name,
            "original_title": original_title,
            "extracted_content_sample": cleaned_text[:1000],
        }

        for m in MODELS:
            col_name = f"title_{m['label']}"
            latency_col = f"latency_sec_{m['label']}"

            print(f"  ├─ Generating with {m['label']} ({m['type']})...")
            gen_title, latency = generate_title_llm(m, cleaned_text)

            row_data[col_name] = gen_title
            row_data[latency_col] = latency

        rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Batch {BATCH_NUM} Complete! Saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
    winsound.MessageBeep(winsound.MB_OK)
