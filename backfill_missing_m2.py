import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util

# Load your embedding model for re-calculating semantic scores
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

PARTITION_FILES = [
    "research_course_summaries_partition_1.csv",
    "research_course_summaries_partition_2.csv",
]



def backfill_partition(csv_path):
    print(f"\n📂 Checking {csv_path} for missing Model 2 scores...")
    df = pd.read_csv(csv_path)
    
    # Identify rows where Model 2 failed (Score is 0.0 or Summary is NaN)
    missing_mask = (df["M2_Semantic_Score"] == 0.0) | (df["M2_Summary"].isna())
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print(" ✅ No missing Model 2 data found in this partition!")
        return

    print(f" ⚠️ Found {missing_count} rows needing backfill for Model 2. Processing...")

    for idx, row in df[missing_mask].iterrows():
        title = row["Course_Title"]
        text = row["Cleaned_Text"]
        ref_summary = row["Reference_Summary"]  # Or whatever your ground truth summary column is named
        
        # Call your updated generate_llm_summary function (with fallback enabled)
        # model_config for Model 2
        m2_config = {"type": "openrouter", "name": "nvidia/nemotron-3-super-120b-a12b:free"}
        
        new_summary, new_latency = generate_llm_summary(
            m2_config, "MODEL_2", title, text, max_retries=4
        )
        
        if new_summary:
            # Re-calculate semantic similarity score
            emb_ref = embed_model.encode(ref_summary, convert_to_tensor=True)
            emb_m2 = embed_model.encode(new_summary, convert_to_tensor=True)
            new_score = round(float(util.cos_sim(emb_ref, emb_m2)[0][0]), 4)
            
            # Update DataFrame
            df.at[idx, "M2_Summary"] = new_summary
            df.at[idx, "M2_Latency"] = new_latency
            df.at[idx, "M2_Semantic_Score"] = new_score
            print(f"  └─ Row {idx+1}: Backfilled successfully! New Score: {new_score}")
        else:
            print(f"  └─ Row {idx+1}: Backfill attempt failed again. Retaining None.")
            
        time.sleep(2) # Polite delay between requests

    # Overwrite CSV with updated values
    df.to_csv(csv_path, index=False)
    print(f" ✅ Saved updated {csv_path}")

# Run backfill for all partitions
for csv_file in PARTITION_FILES:
    try:
        backfill_partition(csv_file)
    except FileNotFoundError:
        print(f" ❌ File not found: {csv_file}")