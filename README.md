# hybrid-medication-extraction

This repository contains a complete pipeline for extracting medication actions from unstructured clinical notes. The system identifies medication mentions and determines whether a clinician is starting, stopping, continuing, holding, or resuming a therapy. It uses a hybrid approach combining rule-based extraction, a biomedical NER model, a curated drug dictionary, and an optional LLM step for resolving ambiguous cases.

The pipeline supports large datasets through chunked processing and parallel LLM execution. All components include logging to support debugging and auditability.

---

## Overview

The project implements the following workflow:

1. **Rule-based action extraction**
   Identifies clinical action phrases such as “start”, “stop”, “continue”, “hold”, “discontinue”, and many variations and shorthands.

2. **Biomedical NER extraction**
   Uses the `d4data/biomedical-ner-all` model to detect medication mentions in text. Runs in batch mode for efficiency.

3. **Dictionary-based identification**
   A drug list (`drug_set.txt`) is loaded and normalized. Matching is performed on cleaned note text to catch medications that the NER model may miss.

4. **Alignment of actions and medications**
   Each action is paired with the closest medication mention based on character indexes.

5. **Confidence scoring**
   Uses the NER confidence score and penalizes disagreements between rule-based and NER detections.

6. **Optional LLM step**
   Low-confidence outputs are refined through an LLM running locally via Ollama.
   Only this subset of results is reviewed to minimize runtime cost.

7. **Chunked execution**
   Notes are processed in chunks (default 100 per chunk).
   NER runs in batch mode, and LLM calls run concurrently.

---

---

## Installation

Clone the repository:

```
git clone https://github.com/div1996/hybrid-medication-extraction.git
cd hybrid-medication-extraction
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows:

```
venv\Scripts\activate
```

macOS/Linux:

```
source venv/bin/activate
```

Install requirements:

```
pip install -r requirements.txt
```

---

## Installing Ollama (Required Only If LLM Features Are Enabled)

Download Ollama:

[https://ollama.com/download](https://ollama.com/download)

After installation, start the Ollama server:

```
ollama serve
```

Download an appropriate model:

```
ollama pull llama3
```

The pipeline will automatically use this model when `--use_llm true` is enabled.

---

## Input Files

### Notes CSV

Expected to contain a column named `note_text`.

Example:

```
note_text
"Patient continues timolol... Start latanoprost..."
"Stop brimonidine due to allergy..."
```

### Drug Dictionary

A text file (`drug_set.txt`) with one drug name per line, cleaned and lowercase.

Example:

```
latanoprost
timolol
brimonidine
aspirin
hydrocodone
```

Place it in the `data/` folder or specify a different path via CLI arguments.

---

## Running the Pipeline

Basic usage with defaults:

```
python medication_extraction.py
```

This uses:

* `Sample_data_to_test.csv`
* `drug_set.txt`
* writes results to `output_notes_df.csv`
* uses chunk size of 100
* LLM enabled

---

## Output

The script produces a CSV with the following columns:

* `note_id`
* `action`
* `keyword`
* `rule_based_med`
* `ner_med`
* `ner_label`
* `confidence`
* `llm_output` (JSON)

Example LLM output inside the cell:

```
{"final_medication": "latanoprost",
 "final_action": "Continue",
 "additional_medications": ["timolol"],
 "rationale": "The note clearly states the patient is continuing latanoprost."}
```

---

## Logging

Logs are written to:

```
logs/med_recon_chunk_parallel.log
```

This file includes:

* Loading of resources
* Chunk processing progress
* Number of detected actions
* LLM call counts
* Errors and fallbacks

---

## Troubleshooting

### Ollama returns 500 or CUDA errors

Reduce the number of workers:

```
--llm_workers 1
```

Restart Ollama:

```
ollama serve
```

### Extremely slow processing

Reduce chunk size:

```
--chunk_size 50
```

Disable LLM:

```
--use_llm false
```

### NER model loads slowly

Install a GPU-enabled PyTorch build or run in CPU-only mode.

---

## Extending the Pipeline

You can extend the project by:

* Adding more action patterns
* Updating or replacing the NER model
* Expanding the drug dictionary
* Introducing new LLM prompts
* Adding dose extraction or temporal reasoning

The code is modular and designed for change.
