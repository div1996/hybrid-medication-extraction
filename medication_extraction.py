import re
import json
import html
import logging
import unicodedata
from typing import List, Dict

import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama

# =====================================================================
# CONFIG
# =====================================================================

CHUNK_SIZE = 100          # number of notes per chunk
LLM_WORKERS = 1           # parallel threads for LLM calls

NOTES_CSV = "Sample_data_to_test.csv"
DRUG_DICT_PATH = "drug_set.txt"
OUTPUT_CSV = "output_notes_df.csv"

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("med_recon_chunk_parallel.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Medication Reconciliation Pipeline (Chunk + Parallel LLM) Initialized.")


# =====================================================================
# SECTION 1 — DRUG DICTIONARY LOADING & NORMALIZATION
# =====================================================================

def clean_drug_name(text: str) -> str:
    """Normalize raw drug names from dictionary or NER output."""
    try:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)                       # Decode HTML entities
        text = re.sub(r"<.*?>", "", text)                # Remove HTML tags
        text = unicodedata.normalize("NFKD", text)       # Normalize Unicode
        text = re.sub(r"[^\w\s\-+()/'\"]", " ", text)    # Keep alphanum + essential chars
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text if len(text) > 2 else ""
    except Exception as e:
        logger.error(f"Drug cleaning failed: {e}")
        return ""


def load_drug_dictionary(path: str) -> List[str]:
    """
    Load cleaned drug names from a text file: one drug per line (already normalized).
    """
    logger.info(f"Loading drug dictionary from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            drug_set = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"Successfully loaded {len(drug_set)} drugs into dictionary.")
        return sorted(drug_set)
    except Exception as e:
        logger.error(f"Failed to load drug dictionary: {e}")
        return []


# =====================================================================
# SECTION 2 — EXTENDED ACTION GRAMMAR
# =====================================================================

START_PATTERNS = [
    r"start", r"started", r"start on", r"started on",
    r"begin", r"beginning", r"initiated", r"initiate",
    r"initiation of", r"commence", r"commenced",
    r"resume", r"resumed", r"restart", r"restarted",
    r"re-initiate", r"reinitiate", r"reinstate",
    r"add", r"added", r"adding",
    r"start taking", r"restart taking",
    r"prescribe", r"prescribed", r"placed on", r"put on",
    r"new rx", r"titrate up", r"increase to", r"uptitrate", r"up-titrate"
]

STOP_PATTERNS = [
    r"stop", r"stopped", r"stop taking",
    r"discontinue", r"discontinued", r"d/c", r"dc",
    r"cease", r"ceased", r"terminate", r"terminated",
    r"hold", r"held", r"withhold", r"withdraw",
    r"drop", r"remove", r"remove from regimen",
    r"taper off", r"wean off", r"reduce to zero",
    r"step down", r"end therapy"
]

CONTINUE_PATTERNS = [
    r"continue", r"continued", r"continuing", r"cont",
    r"maintain", r"maintain on", r"stay on",
    r"keeps taking", r"ongoing medication", r"refill", r"refilled"
]

ACTION_GRAMMAR = START_PATTERNS + STOP_PATTERNS + CONTINUE_PATTERNS


# =====================================================================
# SECTION 3 — RULE-BASED ACTION EXTRACTION
# =====================================================================

def rule_based_extract_actions(text: str) -> List[Dict]:
    """
    Extract action + medication mentions using extended patterns for a single note.
    """
    clean_text = " ".join(text.split())
    actions = []

    for keyword in ACTION_GRAMMAR:
        regex = rf"\b({keyword})\b\s+([A-Za-z0-9\-\_/]+)"
        for match in re.finditer(regex, clean_text, flags=re.IGNORECASE):
            action_raw = match.group(1).lower()
            med_raw = clean_drug_name(match.group(2))

            # Map keyword to canonical action
            if any(re.fullmatch(k, action_raw) for k in START_PATTERNS):
                canonical = "Start"
            elif any(re.fullmatch(k, action_raw) for k in STOP_PATTERNS):
                canonical = "Stop"
            elif any(re.fullmatch(k, action_raw) for k in CONTINUE_PATTERNS):
                canonical = "Continue"
            else:
                canonical = "Unknown"

            actions.append({
                "action": canonical,
                "keyword": action_raw,
                "medication_mention": med_raw,
                "span_start": match.start(),
                "span_end": match.end(),
                "source": "rule_based"
            })

    return actions


def batch_rule_based(notes: List[str]) -> List[List[Dict]]:
    """
    Run rule-based extraction for a list of notes (chunk).
    Returns list-of-lists: one list of actions per note.
    """
    all_actions = []
    for text in notes:
        acts = rule_based_extract_actions(text)
        all_actions.append(acts)
    return all_actions


# =====================================================================
# SECTION 4 — BIOMEDICAL NER EXTRACTION (BATCHED)
# =====================================================================

logger.info("Loading biomedical NER model (first load may take time)...")
NER_MODEL = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
bio_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
logger.info("NER model loaded successfully.")


def batch_biomed_ner(notes: List[str]) -> List[List[Dict]]:
    """
    Run biomedical NER over a chunk of notes.
    Returns list-of-lists: meds[i] is list of med entities for notes[i].
    """
    ents_batch = bio_ner(notes)  # pipeline supports list of strings
    all_meds = []

    for ents in ents_batch:
        meds = []
        for e in ents:
            label = e["entity_group"].upper()
            if any(tag in label for tag in ["DRUG", "CHEM", "MED"]):
                meds.append({
                    "medication_text": clean_drug_name(e["word"]),
                    "start": e["start"],
                    "end": e["end"],
                    "score": float(e["score"]),
                    "label": label,
                    "source": "bio_ner"
                })
        all_meds.append(meds)

    return all_meds


# =====================================================================
# SECTION 5 — DICTIONARY MATCH
# =====================================================================

def batch_dictionary_match(notes: List[str], drug_list: List[str]) -> List[List[Dict]]:
    """
    For each note in the chunk, find dictionary drugs present in the text.
    """
    results = []
    for text in notes:
        text_clean = clean_drug_name(text)
        matches = []
        for drug in drug_list:
            if drug in text_clean:
                matches.append({"medication_text": drug, "source": "dictionary"})
        results.append(matches)
    return results


# =====================================================================
# SECTION 6 — MERGING & ATTACHING MEDS TO ACTIONS
# =====================================================================

def merge_med_lists(*lists):
    """
    Merge multiple medication lists, deduplicating by medication_text.
    """
    merged = {}
    for lst in lists:
        for entry in lst:
            drug = entry["medication_text"]
            if drug not in merged:
                merged[drug] = entry
    return list(merged.values())


def attach_nearest_med(text: str, actions: List[Dict], meds: List[Dict], window: int = 40) -> List[Dict]:
    """
    For each action, attach closest med span in character distance (if any).
    """
    merged_records = []

    for a in actions:
        best = None
        best_dist = 9999
        a_center = (a["span_start"] + a["span_end"]) // 2

        for m in meds:
            if "start" in m and "end" in m:
                m_center = (m["start"] + m["end"]) // 2
                dist = abs(a_center - m_center)
                if dist < best_dist and dist <= window:
                    best = m
                    best_dist = dist

        merged_records.append({
            "text": text,
            "action": a["action"],
            "keyword": a["keyword"],
            "rule_based_med": a["medication_mention"],
            "ner_med": best["medication_text"] if best else None,
            "ner_label": best["label"] if best else None,
            "ner_score": best["score"] if best else None
        })

    return merged_records


# =====================================================================
# SECTION 7 — CONFIDENCE SCORING
# =====================================================================

def score_confidence(record: Dict) -> float:
    """
    Hybrid confidence score:
      - NER score is primary
      - Penalty if rule-based and NER meds differ
    """
    base = record["ner_score"] if record["ner_score"] else 0.55
    if record["rule_based_med"] and record["ner_med"]:
        if record["rule_based_med"] != record["ner_med"]:
            base -= 0.15
    return max(0.0, min(base, 1.0))


# =====================================================================
# SECTION 8 — LLM PROMPT & CALL (PARALLEL PER RECORD)
# =====================================================================

def build_llm_prompt(record: Dict) -> str:
    return f"""
You are an advanced clinical NLP system specializing in medication reconciliation.
Your task is to perform fully comprehensive, medically accurate medication extraction and action interpretation.

Your analysis MUST:
- Use the clinical note as the primary source of truth.
- Correct system mistakes, fill gaps, and recover medications missed by NER or rule-based logic.
- Normalize all medication names and actions with high clinical precision.
- Identify ANY medication, including those not detected by NER or dictionary lookup.

============================================================
CLINICAL NOTE (FULL ORIGINAL TEXT)
============================================================
\"\"\"{record['text']}\"\"\"

============================================================
SYSTEM-DETECTED CANDIDATE INFORMATION (POSSIBLY INCOMPLETE)
============================================================
Rule-based action: {record['action']}
Rule-based medication: {record['rule_based_med']}
NER medication: {record['ner_med']}
NER label: {record['ner_label']}
Confidence score: {record['confidence']:.2f}

============================================================
YOUR TASK — EXTREMELY DETAILED INSTRUCTIONS
============================================================

### STEP 1 — Identify ALL medication names
You MUST:
- Read the entire note carefully and extract every medication mentioned, even if:
  * The rule-based system failed to detect it.
  * The NER model missed it completely.
  * The drug dictionary did not match it.
- Correct incomplete or partial references (e.g., "lata", "timol", "metopro", "brim" → full drug name).
- Normalize:
  * Brand → generic (e.g., "Xalatan" → "latanoprost")
  * Ophthalmic abbreviations (e.g., "PFAT", "AT", "gtts") → normalize to the proper drug
  * Combination drugs into their clinically relevant components when appropriate.
- Include systemic, ophthalmic, OTC, PRN, topical, injectables, supplements (if used clinically).
- Distinguish medications from similarly shaped non-medication terms.

### STEP 2 — Determine the correct action for the PRIMARY medication
For the medication most strongly tied to the detected action:
- Consider explicit action terms (start/stop/continue/hold/resume).
- Consider implicit signals:
  * "back on", "remain on", "to begin", "will restart", "ran out", "needs refill",
    "no longer taking", "stopped using", "discontinued previously", "held for now",
    "increase dose", "titrate up/down".
- Infer the correct standardized action:
  "Start", "Stop", "Continue", "Hold", "Resume", or "Unknown".
- If the action is unclear or conflicting, choose "Unknown".

### STEP 3 — Identify ALL additional medications
You MUST list every medication found anywhere in the note, including:
- Current meds
- Ophthalmic drops
- Discharge medications
- PRN meds
- Allergy-related meds (e.g., "allergic to brimonidine" → include "brimonidine")
- Historical meds if clearly mentioned as medications
- Supplements, vitamins, or herbal compounds ONLY if used with therapeutic intent.

### STEP 4 — Rationale
Provide a concise, clinically sound explanation (1–3 sentences) describing:
- Why you selected the final medication and action.
- Why certain medications were added as additional items.
- How discrepancies between rule-based, NER, dictionary, and text were resolved.

============================================================
RESPONSE FORMAT (STRICT JSON ONLY)
============================================================
{{
  "final_medication": "",
  "final_action": "",
  "additional_medications": [],
  "rationale": ""
}}
"""
def extract_json(text: str) -> str:
    """Extract the first valid JSON object from the text."""
    import re
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, text)
    return matches[0] if matches else None


def llm_review(record: Dict, model: str = "llama3") -> Dict:
    prompt = build_llm_prompt(record)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return {
            "final_medication": record["rule_based_med"] or record["ner_med"],
            "final_action": record["action"],
            "additional_medications": [],
            "rationale": "Fallback — Ollama request crashed."
        }

    # Extract content safely
    content = response.get("message", {}).get("content", "")

    if not content or content.strip() == "":
        logger.error("LLM returned EMPTY content.")
        return {
            "final_medication": record["rule_based_med"] or record["ner_med"],
            "final_action": record["action"],
            "additional_medications": [],
            "rationale": "Fallback — LLM returned empty response."
        }

    # Try extract JSON
    json_text = extract_json(content)
    if json_text:
        try:
            return json.loads(json_text)
        except Exception:
            pass

    # If still failing, log full content for debugging
    logger.error(f"Invalid LLM JSON output: {content[:500]}")

    return {
        "final_medication": record["rule_based_med"] or record["ner_med"],
        "final_action": record["action"],
        "additional_medications": [],
        "rationale": "Fallback — LLM returned invalid JSON."
    }



# =====================================================================
# SECTION 9 — PROCESS A CHUNK OF NOTES
# =====================================================================

def process_chunk(notes_chunk: List[str], note_ids: List[int], drug_list: List[str], use_llm: bool) -> List[Dict]:
    """
    Process a chunk of notes (size up to CHUNK_SIZE):
      - rule-based actions per note
      - NER over all notes in the chunk
      - dictionary matches per note
      - attach actions to meds
      - score confidence
      - optional parallel LLM for low-confidence recs
    """
    # Rule-based per note in chunk
    rb_batch = batch_rule_based(notes_chunk)

    # NER over chunk
    ner_batch = batch_biomed_ner(notes_chunk)

    # Dictionary matches
    dict_batch = batch_dictionary_match(notes_chunk, drug_list)

    # Build candidate records for this chunk
    candidates: List[Dict] = []
    for local_idx, note_text in enumerate(notes_chunk):
        actions = rb_batch[local_idx]
        meds_ner = ner_batch[local_idx]
        meds_dict = dict_batch[local_idx]
        meds_all = merge_med_lists(meds_ner, meds_dict)

        if not actions:
            continue

        merged = attach_nearest_med(note_text, actions, meds_all)
        for rec in merged:
            rec["note_id"] = note_ids[local_idx]
            rec["confidence"] = score_confidence(rec)
            rec["llm"] = None
            candidates.append(rec)

    # Parallel LLM only for low-confidence candidates
    if use_llm:
        low_conf_indices = [i for i, r in enumerate(candidates) if r["confidence"] < 0.75]
        logger.info(f"Chunk with note_ids {note_ids[0]}–{note_ids[-1]}: {len(low_conf_indices)} records need LLM.")

        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
            future_to_idx = {
                ex.submit(llm_review, candidates[i]): i for i in low_conf_indices
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    candidates[i]["llm"] = fut.result()
                except Exception as e:
                    logger.error(f"LLM future error for candidate {i}: {e}")
                    candidates[i]["llm"] = {
                        "final_medication": candidates[i]["rule_based_med"] or candidates[i]["ner_med"],
                        "final_action": candidates[i]["action"],
                        "additional_medications": [],
                        "rationale": "Fallback — LLM call failed."
                    }

    return candidates


# =====================================================================
# SECTION 10 — RUN PIPELINE ON ALL NOTES (CHUNKS)
# =====================================================================

def run_pipeline(notes_csv: str, ligands_txt: str, use_llm: bool = False) -> pd.DataFrame:
    logger.info(f"Loading notes from: {notes_csv}")
    df = pd.read_csv(notes_csv)
    notes = df["note_text"].astype(str).tolist()
    logger.info(f"Total notes loaded: {len(notes)}")

    # Load dictionary
    drug_list = load_drug_dictionary(ligands_txt)

    all_candidates: List[Dict] = []

    # Process in chunks
    n = len(notes)
    logger.info(f"Processing notes in chunks of size {CHUNK_SIZE}...")
    for start in tqdm(range(0, n, CHUNK_SIZE), desc="Chunks"):
        end = min(start + CHUNK_SIZE, n)
        notes_chunk = notes[start:end]
        note_ids = list(range(start, end))

        chunk_candidates = process_chunk(notes_chunk, note_ids, drug_list, use_llm)
        all_candidates.extend(chunk_candidates)

    # Build final DataFrame
    rows = []
    for r in all_candidates:
        rows.append({
            "note_id": r["note_id"],
            "action": r["action"],
            "keyword": r["keyword"],
            "rule_based_med": r["rule_based_med"],
            "ner_med": r["ner_med"],
            "ner_label": r["ner_label"],
            "confidence": r["confidence"],
            "llm_output": r["llm"]
        })

    df_out = pd.DataFrame(rows)
    logger.info("Pipeline completed for all notes.")
    return df_out


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    df_result = run_pipeline(
        notes_csv=NOTES_CSV,
        ligands_txt=DRUG_DICT_PATH,
        use_llm=True
    )
    df_result.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results saved to {OUTPUT_CSV}")
