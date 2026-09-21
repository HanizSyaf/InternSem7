import argparse
import io
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from IPython.display import Markdown, display
import ollama
from openai import OpenAI
import pandas as pd
from PIL import Image
import pymupdf as fitz
import pytesseract
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import winsound
#v13-partition pdf, 4 part

# Load lightweight local embedding model for Semantic Similarity
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

BASE_DIR = Path(r"D:\Intern_Sem7")
CATALOG_PATH = Path("catalog_cards.json")

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================================
# BENCHMARK MODEL CONFIGURATION
# ==========================================
MODEL_1 = {
    "type": "ollama",
    "name": "llama3.2:3b",
}

MODEL_2 = {
    "type": "openrouter",
    "name": "nvidia/nemotron-3-super-120b-a12b:free", 
}

MODEL_3 = {
    "type": "openrouter",
    "name": "deepseek/deepseek-chat",
}

DISABLED_MODELS = {
    "MODEL_1": False,
    "MODEL_2": False,
    "MODEL_3": False,
}

SYSTEM_PROMPT = """You are an expert curriculum marketing specialist.
Your task is to analyze the entire course document and generate a new, engaging 150-word brochure summary.

GUIDELINES:
1. Synthesize core topics across Learning Objectives, Course Outline/Schedule, and Target Audience.
2. Do NOT simply repeat vague overview text; capture the actual syllabus content and practical takeaways.
3. IGNORE ALL NOISE: Omit company contact details, website links, HRDF claimable badges, company reviews, break times, and venue/logistical details.
4. STRICT FORMAT: Plain text only (no markdown bolding, headers, or bullet points).
5. LENGTH: Approximately 150 words."""

openrouter_client = OpenAI(
    base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY
)

OVERVIEW_START_PATTERNS = [
    r"program\s+overview",
    r"course\s+overview",
    r"overview",
    r"why\s+matters",
    r"about\s+this\s+course",
    r"introduction",
    r"course\s+description",
]
OVERVIEW_END_PATTERNS = [
    r"learning\s+objective[s]?",
    r"workshop\s+objective[s]?",
    r"course\s+objective[s]?",
    r"program\s+outline",
    r"course\s+schedule",
    r"pre-requisite[s]?",
    r"prerequisite[s]?",
    r"learning\s+outcome[s]?",
    r"expected\s+outcome[s]?",
    r"target\s+audience",
    r"duration",
]


def load_catalog(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_spaced_text(text: str) -> str:
    return re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", text)


def clean_pdf_noise(raw_text: str) -> str:
    text = fix_spaced_text(raw_text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
    )
    text = re.sub(
        r"(?i)100%\s+hrdf\s+claimable|hrdcorp\s+claimable|registered\s+hrdcorp\s+training\s+provider",
        "",
        text,
    )
    text = re.sub(r"(?i)about\s+elite\s+indigo.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)why\s+choose\s+us\?.*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n+", "\n", text).strip()


def extract_page_text_ordered(page) -> str:
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (round(b[1], -1), b[0]))
    page_lines = [b[4].strip() for b in blocks if b[4].strip()]
    return "\n".join(page_lines)


def extract_targeted_overview(full_text: str) -> str:
    start_regex = (
        r"(?i)(?:[•\-*\s]*)(?:" + "|".join(OVERVIEW_START_PATTERNS) + r")"
    )
    end_regex = r"(?i)(?:[•\-*\s]*)(?:" + "|".join(OVERVIEW_END_PATTERNS) + r")"

    start_match = re.search(start_regex, full_text)
    if not start_match:
        paragraphs = [
            p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 60
        ]
        return (
            paragraphs[0]
            if paragraphs
            else full_text[:500].replace("\n", " ").strip()
        )

    start_idx = start_match.end()
    text_after_start = full_text[start_idx:]

    text_after_start = re.sub(
        r"^(?:Learning Objectives|Duration|Course Schedule|Program Outline)\s*",
        "",
        text_after_start,
        flags=re.IGNORECASE,
    )

    end_match = re.search(end_regex, text_after_start)
    clean_overview = (
        text_after_start[: end_match.start()]
        if end_match
        else text_after_start[:1000]
    )
    clean_overview = re.sub(r"\s+", " ", clean_overview).strip()

    if len(clean_overview) < 30:
        clean_overview = re.sub(r"\s+", " ", text_after_start[:800])
        clean_overview = re.sub(
            r"(?i)Learning Objectives.*", "", clean_overview
        ).strip()

    return clean_overview


def extract_raw_schedule_block(full_text: str) -> str:
    start_patterns = [
        r"(?i)module\s+1",
        r"(?i)course\s+schedule",
        r"(?i)program\s+outline",
        r"(?i)course\s+outline",
        r"(?i)morning\s+session",
        r"(?i)afternoon\s+session",
        r"(?i)agenda",
        r"(?i)games\s*&\s*activities",
    ]

    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, full_text)
        if match:
            start_idx = match.start()
            break

    raw = full_text[start_idx : start_idx + 6000]
    raw = re.split(
        r"(?i)(about\s+elite\s+indigo|contact\s+us|for\_more\_100%_hrdf|testimonials)",
        raw,
    )[0]

    noise_patterns = [
        r"(?i)DAY\s*/\s*TIME\s*DESCRIPTION",
        r"(?i)\d+\s*(Hours?|Mins?|Minutes?)\b",
        r"(?i)OVERALL\s+TIME\s*:\s*ABOUT\s*\d+.*?\n",
        r"(?i)Opening\s*&\s*Recap",
        r"(?i)Summary\s*and\s*End\s*of\s*Day\s*\d+",
        r"(?i)Continued\s+on\s+Next\s+Page",
        r"(?i)Lunch(\s*break)?",
        r"(?i)Buffet\s+Lunch\s+is\s+served\.?",
    ]
    for pattern in noise_patterns:
        raw = re.sub(pattern, " ", raw)

    return re.sub(r"\n\s*\n", "\n", raw).strip()


def format_outline_with_llm(raw_schedule_text: str) -> str:
    if not raw_schedule_text or len(raw_schedule_text) < 20:
        return raw_schedule_text

    formatting_prompt = """You are a precise syllabus parser. Transform raw OCR course schedule text into clean, structured Markdown.

CRITICAL INSTRUCTION: Analyze the raw text and select the MOST SUITABLE output format:

--- FORMAT OPTION 1: IF THE PDF USES MODULES ---
Duration: X Days

### Day 1:
* Module 1: Title
  - Sub-topic description

--- FORMAT OPTION 2: IF THE PDF USES SESSIONS ---
Duration: X Day(s)

### Morning Session:
* MAIN TOPIC / WORKSHOP TITLE
  - Sub-topic or activity description

--- STRICT RULES ---
1. Include ALL topics from source text.
2. Omit 'Registration', 'Break', 'Lunch', and page footers.
3. Fix concatenated OCR words."""

    try:
        response = ollama.chat(
            model=MODEL_1["name"],
            messages=[
                {"role": "system", "content": formatting_prompt},
                {
                    "role": "user",
                    "content": f"Raw Schedule Text:\n{raw_schedule_text}",
                },
            ],
            options={"num_predict": 1500, "temperature": 0.0},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama formatting error: {e}")
        return raw_schedule_text


def extract_text_from_page_images(page) -> str:
    ocr_text = []
    image_list = page.get_images(full=True)

    if image_list:
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                extracted = pytesseract.image_to_string(image).strip()
                if extracted:
                    ocr_text.append(extracted)
            except Exception:
                continue

    if not ocr_text:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text.append(pytesseract.image_to_string(img).strip())

    return "\n".join(ocr_text).strip()


def extract_pdf_data(pdf_path: Path) -> tuple[str, str, str, str, float]:
    start_time = time.time()
    doc = fitz.open(pdf_path)
    full_text = []
    ocr_page_count = 0
    total_pages = len(doc)

    schedule_keywords = [
        "module", "schedule", "outline", "day 1",
        "agenda", "session", "topic", "morning", "afternoon"
    ]

    for page in doc:
        text = extract_page_text_ordered(page)

        if len(text) <= 50:
            ocr_page_count += 1
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            full_text.append(pytesseract.image_to_string(img))
        else:
            has_schedule_kw = any(
                kw in text.lower() for kw in schedule_keywords
            )
            image_list = page.get_images(full=True)

            if not has_schedule_kw and len(image_list) > 0:
                ocr_img_text = extract_text_from_page_images(page)
                if ocr_img_text:
                    text += f"\n\n--- [IMAGE OCR CONTENT] ---\n{ocr_img_text}"

            full_text.append(text)

    pdf_type = "img" if (ocr_page_count / max(total_pages, 1)) > 0.5 else "txt"
    raw_combined = "\n\n".join(full_text)
    clean_combined = clean_pdf_noise(raw_combined)

    original_overview = extract_targeted_overview(clean_combined)
    raw_schedule = extract_raw_schedule_block(clean_combined)
    outline_summary = format_outline_with_llm(raw_schedule)

    extraction_time = round(time.time() - start_time, 2)
    return (
        clean_combined,
        pdf_type,
        original_overview,
        outline_summary,
        extraction_time,
    )


def calculate_semantic_similarity(summary, reference_text):
    # Safe handling: Return 0.0 score for failed LLM runs
    if not summary or summary == "Null":
        return 0.0

    emb1 = semantic_model.encode(summary, convert_to_tensor=True)
    emb2 = semantic_model.encode(reference_text, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round(score, 4)


def match_pdf_to_catalog(pdf_name: str, catalog: list) -> dict:
    clean_name = Path(pdf_name).stem
    titles = [item["title"] for item in catalog]
    best_match, score, index = process.extractOne(
        clean_name, titles, scorer=fuzz.token_sort_ratio
    )
    return catalog[index] if score > 60 else None


def generate_llm_summary(
    model_config: dict,
    model_key: str,
    course_title: str,
    full_cleaned_text: str,
    max_retries: int = 2,
) -> tuple[str | None, float]:
    """
    Generates LLM summary with a retry loop for temporary errors and 
    circuit-breaking (DISABLED_MODELS) for quota/rate limits.
    """
    # 1. Early exit if the model was flagged as quota-exceeded in a previous run
    if DISABLED_MODELS.get(model_key, False):
        return None, 0.0

    prompt = f"Course Title: {course_title}\n\nFull Course Document Content:\n{full_cleaned_text[:8000]}"
    start_time = time.time()

    for attempt in range(1, max_retries + 1):
        try:
            if model_config["type"] == "ollama":
                response = ollama.chat(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                # Safe dict extraction
                summary = response.get("message", {}).get("content", "").strip()

            elif model_config["type"] == "openrouter":
                response = openrouter_client.chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                # Safe choice extraction
                if response and hasattr(response, "choices") and response.choices:
                    summary = response.choices[0].message.content.strip()
                else:
                    raise ValueError("Empty payload returned from OpenRouter.")

            latency = round(time.time() - start_time, 2)
            return summary, latency

        except Exception as e:
            err_msg = str(e).lower()

            # Check if error is unrecoverable (quota/credits) vs transient
            is_quota_error = any(
                kw in err_msg
                for kw in [
                    "insufficient_quota",
                    "rate_limit",
                    "429",
                    "credit",
                    "balance",
                    "quota",
                ]
            )

            if is_quota_error:
                print(
                    f"    ⚠️ Quota/Rate limit reached for {model_key} -> {type(e).__name__}: {e}. Skipping model for remaining runs."
                )
                DISABLED_MODELS[model_key] = True
                break  # Stop retrying if the quota is exceeded

            # Log attempt failure for transient errors
            print(
                f"    ⚠️ [{model_key}] Attempt {attempt}/{max_retries} Failed -> {type(e).__name__}: {e}"
            )

            # Delay before retrying transient failures
            if attempt < max_retries:
                time.sleep(3)

    # Return None if retries were exhausted or a quota error occurred
    latency = round(time.time() - start_time, 2)
    print(f"    ❌ [{model_key}] Execution failed. Defaulting to None.")
    return None, latency

def get_model_identifier(model_config: dict) -> str:
    return model_config["name"].replace(":", "_").replace("/", "_")


def main():
    start_partition_time = time.time()

    parser = argparse.ArgumentParser(
        description="Process a single partition batch of course PDFs."
    )
    parser.add_argument(
        "--partition",
        type=int,
        choices=[1, 2, 3, 4],
        default=int(os.getenv("PARTITION", 1)),
        help="Partition index to process (1, 2, 3, or 4)",
    )
    args = parser.parse_args()
    partition_id = args.partition

    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY missing in .env file.")
        return

    # NEW LOGIC (Matches your folder structure directly)
    target_folder = BASE_DIR / f"INDIGO courses_P{partition_id}"
    if not target_folder.exists():
        print(f"❌ Error: Directory not found -> {target_folder}")
        return

    pdf_files = list(target_folder.rglob("*.pdf"))

    output_csv = Path(f"research_course_summaries_partition_{partition_id}.csv")
    catalog = load_catalog(CATALOG_PATH)
    rows = []

    # FIXED: Replaced len(subfolders) with total partition count (4)
    print(
        f"🚀 Running Partition {partition_id}/4: Target Folder -> '{target_folder.name}'"
    )
    print(f"Found {len(pdf_files)} PDFs to process. Output target: {output_csv}\n")

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")

        category_folder = pdf_file.parent.name
        matched = match_pdf_to_catalog(pdf_file.name, catalog)

        if not matched:
            print("  └─ ⚠️ Skipped (No catalog match)")
            continue

        try:
            (
                cleaned_text,
                pdf_type,
                orig_overview,
                outline_summary,
                ocr_time,
            ) = extract_pdf_data(pdf_file)

            print("  └─ Running Model 1 (Ollama)...")
            sum1, time1 = generate_llm_summary(
                MODEL_1, "MODEL_1", matched["title"], cleaned_text
            )

            print("  └─ Running Model 2 (OpenRouter)...")
            sum2, time2 = generate_llm_summary(
                MODEL_2, "MODEL_2", matched["title"], cleaned_text
            )

            print("  └─ Running Model 3 (OpenRouter Paid)...")
            sum3, time3 = generate_llm_summary(
                MODEL_3, "MODEL_3", matched["title"], cleaned_text
            )

            print("  └─ Calculating Semantic Similarity Scores...")
            rich_reference_text = (
                f"{orig_overview}\n\nSyllabus Outline:\n{outline_summary}"
            )

            sem1 = calculate_semantic_similarity(sum1, rich_reference_text)
            sem2 = calculate_semantic_similarity(sum2, rich_reference_text)
            sem3 = calculate_semantic_similarity(sum3, rich_reference_text)

            # --- LIVE TERMINAL OUTPUT ---
            print(f"  └─ Latency (s)  -> M1: {time1}s | M2: {time2}s | M3: {time3}s")
            print(f"  └─ Semantic     -> M1: {sem1} | M2: {sem2} | M3: {sem3}")

            m1_tag = get_model_identifier(MODEL_1)
            m2_tag = get_model_identifier(MODEL_2)
            m3_tag = get_model_identifier(MODEL_3)

            rows.append(
                {
                    "id": matched["course_id"],
                    "category": category_folder,
                    "title": matched["title"],
                    "pdf_type": pdf_type,
                    "original_overview": orig_overview,
                    "outline_summary": outline_summary,
                    "extraction_time_sec": ocr_time,
                    f"summary_{m1_tag}": sum1,
                    f"summary_{m2_tag}": sum2,
                    f"summary_{m3_tag}": sum3,
                    f"latency_sec_{m1_tag}": time1,
                    f"latency_sec_{m2_tag}": time2,
                    f"latency_sec_{m3_tag}": time3,
                    f"semantic_score_{m1_tag}": sem1,
                    f"semantic_score_{m2_tag}": sem2,
                    f"semantic_score_{m3_tag}": sem3,
                }
            )

        except Exception as e:
            print(f"  └─ ❌ Error on file {pdf_file.name}: {e}")

        # Pause 2 seconds between runs to prevent OpenRouter rate limiting
        time.sleep(2)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    # Calculate elapsed time
    total_sec = round(time.time() - start_partition_time, 2)
    mins = int(total_sec // 60)
    secs = round(total_sec % 60, 1)
    
    print(f"\n Partition {partition_id} finished in {mins}m {secs}s! Saved to '{output_csv}'.")


if __name__ == "__main__":
    main()

winsound.MessageBeep(winsound.MB_OK)