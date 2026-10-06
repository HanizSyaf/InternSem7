"""
Multi-LLM Title Generation Pipeline v3 (Production Reliability Edition)
- Target: >99% Success Rate across 840 generations (168 courses x 5 models)
- Features: Exponential backoff, Model Fallback Chains, Resumable Execution (Caching),
  Dedicated OpenRouter Rate Limiting, Strict Structural Validation, and Deep Audit Logs.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Directories & Configuration
REPO_DIR = Path(r"D:\InternSem7-test\restart\extracted_repository")
OUTPUT_CSV = Path("multi_llm_generated_titles_v3.csv")
PILOT_LIMIT = 5  # Set to integer (e.g., 5) for pilot dry run, or None for all 168 courses

# API Clients
ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
)

openrouter_client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# OpenRouter Models
OPENROUTER_FREE_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_PAID_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert curriculum design specialist.
Synthesize the provided course overview and learning objectives into a professional, compelling Title Case course title.

STRICT OPERATING RULES:
1. DIRECT OUTPUT: Output ONLY the final title string. Do NOT include reasoning, thinking steps, commentary, quotes, or markdown.
2. NO COPYING: Do NOT copy phrases or sentence fragments verbatim from the text.
3. CAPITALIZATION: Capitalize in Title Case (preserve acronyms like ESG, AI, IT, HR, QA, ISO, SOP, KPI, ROI).
4. LENGTH: Keep the title strictly between 4 and 12 words.
"""

KNOWN_ACRONYMS = {
    "Esg": "ESG", "Ai": "AI", "It": "IT", "Hr": "HR", "Qa": "QA", "Qc": "QC",
    "Kpi": "KPI", "Iso": "ISO", "Sop": "SOP", "Roi": "ROI", "Fci": "FCI",
    "Hvac": "HVAC", "Plc": "PLC", "Scada": "SCADA", "B2B": "B2B", "Ehs": "EHS", "Osha": "OSHA"
}

GENERIC_BLOCKED_TITLES = {
    "title", "course title", "generated title", "n/a", "unknown", "course outline",
    "proposed title", "untitled course", "untitled", "learning outcomes", "overview"
}


def sanitize_payload_text(text: str) -> str:
    """Removes course codes, headers, and title leakage phrases from prompt text."""
    if not text:
        return ""
    text = re.sub(r"(course title|course outline|outline|welcome to)\s*:\s*.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z]{2,4}-\d{2,4}\b", "", text)
    return text.strip()


def clean_and_validate_title(raw: str) -> str | None:
    """Strips thinking tags, conversational prefixes, quotes, and enforces valid non-generic text."""
    if not raw:
        return None
    
    # 1. Strip reasoning blocks (Qwen <think>...</think>)
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    
    # 2. Strip conversational prefixes, quotes, markdown, bullet markers
    clean = re.sub(r"^(here is|proposed title|course title|title|generated title)\s*:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r'^[#*"`\s\d\.-]+|[#*"`\s]+$', "", clean.strip())
    
    # 3. Validation: Reject Empty, Extremely Short/Long, or Generic Non-Titles
    words = clean.split()
    word_count = len(words)
    if word_count < 3 or word_count > 15:
        return None
    if clean.lower() in GENERIC_BLOCKED_TITLES:
        return None

    # 4. Capitalization & Acronym Formatting
    formatted = [KNOWN_ACRONYMS.get(w.capitalize(), w.capitalize()) for w in words]
    res = " ".join(formatted)
    for lower, upper in KNOWN_ACRONYMS.items():
        res = re.sub(rf"\b{lower}\b", upper, res)
    return res


def execute_single_request(
    client: OpenAI, model_name: str, prompt: str, timeout: int = 120
) -> tuple[str | None, str, str | None]:
    """Executes a single API completion call with timeout and returns (cleaned_title, raw_response, error_str)."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=timeout,
        )
        raw_out = response.choices[0].message.content or ""
        cleaned = clean_and_validate_title(raw_out)
        if cleaned:
            return cleaned, raw_out, None
        return None, raw_out, "ValidationFailed: EmptyOrInvalidFormat"
    except Exception as e:
        err_msg = str(e).split("\n")[0]
        return None, "", f"APIError: {err_msg[:60]}"


def query_model_with_adaptive_retry(
    primary_client: OpenAI,
    primary_model: str,
    repo_data: dict,
    fallback_client: OpenAI = None,
    fallback_model: str = None,
    is_free_cloud: bool = False,
    max_retries: int = 4,
) -> dict:
    """
    Queries a target model with Adaptive Exponential Backoff retries.
    If all retries fail, routes execution through the Fallback Model chain.
    """
    overview = sanitize_payload_text(repo_data.get('overview', ''))
    objectives = sanitize_payload_text(', '.join(repo_data.get('objectives', [])))
    outcomes = sanitize_payload_text(', '.join(repo_data.get('learning_outcomes', [])))
    audience = sanitize_payload_text(repo_data.get('target_audience', ''))
    keywords = sanitize_payload_text(', '.join(repo_data.get('keywords', [])))

    content_summary = f"Overview: {overview}\nObjectives: {objectives}\nLearning Outcomes: {outcomes}\nTarget Audience: {audience}\nKeywords: {keywords}".strip()
    prompt = f"Course Intelligence Context:\n\n{content_summary}"

    start_time = time.time()
    attempt_logs = []
    last_raw_response = ""
    last_error = ""

    # Primary Attempt Loop with Exponential Backoff
    for attempt in range(1, max_retries + 1):
        # Priority 3: Exponential Backoff (2s, 4s, 8s, 16s)
        if attempt > 1:
            backoff = 2 ** (attempt - 1)
            attempt_logs.append(f"Retry Backoff Sleep ({backoff}s)")
            time.sleep(backoff)

        title, raw_resp, err = execute_single_request(primary_client, primary_model, prompt)
        if raw_resp:
            last_raw_response = raw_resp

        if title:
            attempt_logs.append(f"Attempt {attempt}: Success")
            return {
                "title": title,
                "requested_model": primary_model,
                "actual_model_used": primary_model,
                "fallback_used": False,
                "attempts": attempt,
                "latency_sec": round(time.time() - start_time, 2),
                "error_type": "None",
                "attempt_history": " || ".join(attempt_logs),
                "raw_response": last_raw_response,
            }
        
        last_error = err or "UnknownError"
        attempt_logs.append(f"Attempt {attempt} Failed [{last_error}]")

    # Priority 1: Fallback Chain Triggered
    if fallback_client and fallback_model:
        attempt_logs.append(f"Triggering Fallback Model -> {fallback_model}")
        fb_title, fb_raw, fb_err = execute_single_request(fallback_client, fallback_model, prompt)
        
        if fb_title:
            attempt_logs.append("Fallback Succeeded")
            return {
                "title": fb_title,
                "requested_model": primary_model,
                "actual_model_used": fallback_model,
                "fallback_used": True,
                "attempts": max_retries + 1,
                "latency_sec": round(time.time() - start_time, 2),
                "error_type": f"PrimaryFailed_FallbackSuccess ({last_error})",
                "attempt_history": " || ".join(attempt_logs),
                "raw_response": fb_raw,
            }

    # Final Guardrail: Emergency Flagging for Manual Review (Never throw unhandled exceptions)
    attempt_logs.append("FLAGGED_FOR_MANUAL_REVIEW")
    return {
        "title": "FLAG_MANUAL_REVIEW",
        "requested_model": primary_model,
        "actual_model_used": "NONE",
        "fallback_used": True,
        "attempts": max_retries + 1,
        "latency_sec": round(time.time() - start_time, 2),
        "error_type": f"AllAttemptsFailed ({last_error})",
        "attempt_history": " || ".join(attempt_logs),
        "raw_response": last_raw_response,
    }


def load_existing_cache() -> dict[str, dict]:
    """Priority 4: Loads cached rows from existing CSV to skip completed courses."""
    if not OUTPUT_CSV.exists():
        return {}
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if df.empty or "course_id" not in df.columns:
            return {}
        
        # Verify complete valid rows
        cache = {}
        for _, row in df.iterrows():
            cid = row["course_id"]
            # Consider cached if all 5 model titles exist and are not flagged as failed
            title_cols = [c for c in df.columns if c.startswith("title_")]
            if len(title_cols) >= 5 and all(pd.notnull(row[c]) and row[c] != "Generation Failed" for c in title_cols):
                cache[cid] = row.to_dict()
        return cache
    except Exception as e:
        print(f"⚠️ Cache Load Warning: {e}")
        return {}


def process_course(idx: int, total: int, filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cid = data.get("course_id", filepath.stem)
    catalog_title = data.get("catalog_title", "N/A")

    print(f"[{idx}/{total}] Processing ID: {cid} | Original: {catalog_title[:40]}...")

    # Priority 2 & 7: Rate-Limited Cloud Tier (Sequential execution for Free Tier)
    # 1. OpenRouter Free Tier (Gemma 31B) with Fallback to GPT-4o-Mini
    print("  ├─ [Tier B - Cloud Free] Querying google/gemma-4-31b-it:free...")
    res_free = query_model_with_adaptive_retry(
        openrouter_client, OPENROUTER_FREE_MODEL, data,
        fallback_client=openrouter_client, fallback_model=OPENROUTER_PAID_MODEL,
        is_free_cloud=True, max_retries=4
    )
    time.sleep(3.0)  # Priority 7: Inter-request throttle for free endpoint

    # 2. OpenRouter Paid Tier (GPT-4o-Mini)
    print("  ├─ [Tier A - Cloud Paid] Querying openai/gpt-4o-mini...")
    res_paid = query_model_with_adaptive_retry(
        openrouter_client, OPENROUTER_PAID_MODEL, data,
        fallback_client=ollama_client, fallback_model="qwen2.5:7b",
        max_retries=3
    )

    # 3. Local Sequential Execution (Protect VRAM)
    print("  ├─ [Tier A - Local] Querying gemma3:4b...")
    res_gemma_4b = query_model_with_adaptive_retry(ollama_client, "gemma3:4b", data, max_retries=2)

    print("  ├─ [Tier A - Local] Querying qwen2.5:7b...")
    res_qwen_7b = query_model_with_adaptive_retry(ollama_client, "qwen2.5:7b", data, max_retries=2)

    print("  ├─ [Tier C - Local Slow] Querying qwen3.8:27b...")
    res_qwen_27b = query_model_with_adaptive_retry(
        ollama_client, "qwen3.8:27b", data,
        fallback_client=ollama_client, fallback_model="qwen2.5:7b",
        max_retries=2
    )

    # Assemble Audit Record with Priority 9 Reliability Metrics
    record = {
        "course_id": cid,
        "catalog_title": catalog_title,
        "cover_page_title": data.get("cover_page_title", ""),
        "pdf_metadata_title": data.get("pdf_metadata_title", ""),
        "filename_title": data.get("filename_title", ""),

        # Titles
        "title_gemma3_4b": res_gemma_4b["title"],
        "title_qwen2_5_7b": res_qwen_7b["title"],
        "title_qwen3_27b": res_qwen_27b["title"],
        "title_openrouter_free": res_free["title"],
        "title_openrouter_paid": res_paid["title"],

        # Latencies
        "latency_gemma3_4b": res_gemma_4b["latency_sec"],
        "latency_qwen2_5_7b": res_qwen_7b["latency_sec"],
        "latency_qwen3_27b": res_qwen_27b["latency_sec"],
        "latency_openrouter_free": res_free["latency_sec"],
        "latency_openrouter_paid": res_paid["latency_sec"],

        # Priority 9: Reliability Metrics & Audit Trails
        "actual_model_openrouter_free": res_free["actual_model_used"],
        "fallback_used_openrouter_free": res_free["fallback_used"],
        "attempts_openrouter_free": res_free["attempts"],
        "error_type_openrouter_free": res_free["error_type"],
        
        "actual_model_openrouter_paid": res_paid["actual_model_used"],
        "fallback_used_openrouter_paid": res_paid["fallback_used"],
        "attempts_openrouter_paid": res_paid["attempts"],
        "error_type_openrouter_paid": res_paid["error_type"],

        "raw_response_openrouter_free": res_free["raw_response"],
        "raw_response_openrouter_paid": res_paid["raw_response"],
    }

    print("  └─ ✅ Course completed successfully.\n")
    return record


def main():
    json_files = sorted(list(REPO_DIR.glob("*.json")))
    if not json_files:
        print("❌ No extracted JSON files found!")
        return

    target_files = json_files[:PILOT_LIMIT] if PILOT_LIMIT else json_files
    
    # Priority 4: Cache Checking
    cache = load_existing_cache()
    print("=" * 75)
    print(f"🚀 Multi-LLM Generation Pipeline v3 (Target: >99% Success)")
    print(f"📂 Total Courses Found: {len(target_files)} | Cached/Skipped: {len(cache)}")
    print("=" * 75 + "\n")

    records = list(cache.values())
    uncached_files = [f for f in target_files if f.stem not in cache and json.load(open(f)).get("course_id") not in cache]

    start_time = time.time()

    for idx, filepath in enumerate(uncached_files, 1):
        rec = process_course(idx, len(uncached_files), filepath)
        records.append(rec)

        # Incremental CSV Auto-Save after every single course
        df_temp = pd.DataFrame(records)
        df_temp.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total_time = round(time.time() - start_time, 2)
    print("=" * 75)
    print(f"🏆 COMPLETE: Processed {len(records)} records in {total_time}s")
    print(f"💾 File updated: '{OUTPUT_CSV.name}'")
    print("=" * 75)


if __name__ == "__main__":
    main()