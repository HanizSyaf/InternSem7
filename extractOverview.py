# %%
import io
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import pymupdf as fitz
import pytesseract
from rapidfuzz import fuzz, process

load_dotenv()

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
PDF_DIR = Path(r"D:\Intern_Sem7\INDIGO courses")
CATALOG_PATH = Path(r"D:\Intern_Sem7\catalog_cards.json")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

OPENROUTER_MODEL = "openrouter/free"

TARGET_IDS = [
    "c-55de08e6a5",
    "c-493724d2f6",
    "c-9375dddbf8",
    "c-320401822c",
    "c-919c5b7891",
    "c-e99b19e372",
    "c-b2cbde9198",
    "c-f88c90d09c",
    "c-8b999366c1",
    "c-a6d4b241a8",
]

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

EXTRACTION_SYSTEM_PROMPT = """You are an expert curriculum parser. 
Extract information from the raw course text into strict JSON format.

RULES FOR EXTRACTION:

1. OVERVIEW:
   - Extract ONLY the core high-level course description/summary.
   - Strip contact info, websites, HRDF details, footers, and noise characters.
   - STRICT RULE: IF NO OVERVIEW EXISTS, OR IF THE TEXT CONTAINS MODULE LISTS, SCHEDULES, OR TIME BLOCKS (e.g. '3 Hours', 'Day 1', 'Lunch'), RETURN null.

2. OUTLINE:
   - Determine training DURATION: e.g., "1", "2", or "Not Specified".
   - Classify OUTLINE FORMAT TYPE as exactly one of:
       * "A": Module-based WITH sub-descriptions/bullets (e.g., Module 1: Title -> - subtopic).
       * "B": Module-based WITHOUT sub-descriptions/bullets (e.g., Module 1: Title, Module 2: Title).
       * "C": Session-based (MUST contain explicit headers like 'Morning Session' or 'Afternoon Session') WITH sub-descriptions.
       * "D": Session-based (MUST contain explicit headers like 'Morning Session' or 'Afternoon Session') WITHOUT sub-descriptions.
       * "E": Freeform activity list, time blocks ('3 Hours'), or general agenda flow (use this if it fits none of the above).
   - DEDUPLICATE HEADERS: Never repeat module/session titles redundantly.
   - Extract outline items as an array of structured strings matching the format.

STRICT JSON OUTPUT FORMAT (No markdown wrappers):
{
  "overview": "Clean overview string or null",
  "duration": "e.g., '2' or 'Not Specified'",
  "format_type": "A | B | C | D | E",
  "outline_items": [
    "Item heading or bullet matching the rules above"
  ]
}"""


def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def global_noise_cleaner(text: str) -> str:
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
    text = re.sub(
        r"(?i)(about\s+elite\s+indigo|why\s+choose\s+us\?|testimonials|contact\s+us).*",
        "",
        text,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_raw_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        text = page.get_text("text").strip()
        if len(text) < 50:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
        full_text.append(text)

    return "\n\n".join(full_text)


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
    """Python guardrail to reject schedule leakage or low-quality overview snippets."""
    if not overview or not isinstance(overview, str):
        return None
    
    clean_ov = overview.strip()
    if len(clean_ov) < 35:
        return None

    # Reject if overview contains clear schedule markers
    schedule_indicators = [
        r"\bmodule\s+\d+",
        r"\bday\s+\d+",
        r"\b\d+\s+hours?\b",
        r"\blunch\b",
        r"\bmorning\s+session\b",
        r"\bafternoon\s+session\b"
    ]
    for pattern in schedule_indicators:
        if re.search(pattern, clean_ov, re.IGNORECASE):
            return None

    return clean_ov


def sanitize_and_parse_json(raw_text: str) -> dict:
    """Extracts JSON structure even if LLM wraps it in extra characters or text."""
    clean_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"^```\s*", "", clean_text)
    clean_text = re.sub(r"\s*```$", "", clean_text).strip()

    # Find first { and last }
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    
    if start_idx != -1 and end_idx != -1:
        clean_text = clean_text[start_idx : end_idx + 1]

    return json.loads(clean_text)


def run_llm_extraction(cleaned_text: str, course_title: str, max_retries: int = 3) -> dict:
    prompt = f"Course Title: {course_title}\n\nDocument Text:\n{cleaned_text[:7000]}"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            raw_out = response.choices[0].message.content
            if not raw_out:
                raise ValueError("Empty response payload from LLM.")

            parsed_json = sanitize_and_parse_json(raw_out)
            
            # Apply Python overview validation
            parsed_json["overview"] = validate_overview(parsed_json.get("overview"))
            return parsed_json

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "overview": None,
                "duration": "Error",
                "format_type": "E",
                "outline_items": [f"Parsing Error after {max_retries} attempts: {e}"],
            }


def main():
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY is missing in your .env file.")
        return

    print("🔍 Loading catalog and scanning PDF directory...")
    catalog = load_catalog(CATALOG_PATH)
    catalog_dict = {item["course_id"]: item for item in catalog}

    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs. Extracting target IDs via OpenRouter...\n")

    processed_count = 0

    for target_id in TARGET_IDS:
        matched_item = catalog_dict.get(target_id)
        if not matched_item:
            print(f"⚠️ Target ID {target_id} not found in catalog. Skipping.")
            continue

        target_title = matched_item["title"]
        matched_pdf = find_matching_pdf(target_id, target_title, pdf_files)

        if not matched_pdf:
            print(f"❌ Could not find matching PDF for ID: {target_id} | Title: '{target_title}'")
            continue

        processed_count += 1
        print("=" * 80)
        print(f"[{processed_count}/{len(TARGET_IDS)}] ID: {target_id} | Title: {target_title}")
        print(f"File : {matched_pdf.name}")
        print("=" * 80)

        raw_text = extract_raw_pdf_text(matched_pdf)
        cleaned_text = global_noise_cleaner(raw_text)

        result = run_llm_extraction(cleaned_text, target_title)

        print("\n🔹 OVERVIEW:")
        if result.get("overview"):
            print(result["overview"])
        else:
            print("null (No valid overview detected)")

        print("\n🔹 OUTLINE INFO:")
        print(f"  • Duration    : {result.get('duration', 'Not Specified')}")
        print(f"  • Format Type : {result.get('format_type', 'E')}")

        print("\n🔹 OUTLINE CONTENT:")
        items = result.get("outline_items", [])
        if isinstance(items, list):
            for item in items:
                print(f"  * {item}")
        else:
            print(items)
            
        print("\n" + "-" * 80 + "\n")
        time.sleep(1)


if __name__ == "__main__":
    main()