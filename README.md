# hybrid-medication-extraction

This repository provides an end-to-end pipeline for extracting medication-related actions from unstructured clinical notes. The system processes free-text medical documentation to identify medication mentions and determine whether the clinician is starting, stopping, or continuing a drug therapy. The approach integrates rule-based extraction, a biomedical NER model, a curated drug dictionary, and an optional language model for resolving ambiguous cases.

The pipeline is designed for large-scale processing tasks and supports batched execution, chunking, and parallel LLM calls. All modules include logging for traceability.

Core Features

Rule-based action extraction using an extended clinical grammar.

Transformer-based biomedical NER for medication recognition.

Dictionary-based detection to catch medications not identified by the model.

Medication–action alignment using character proximity.

Confidence scoring with penalties for disagreements between subsystems.

Optional LLM refinement for uncertain predictions via Ollama.

Batch and chunk processing for efficient scaling.

Parallel LLM execution to accelerate slow components.

Command-line configuration with fallback to default parameters.

Architecture Overview

The system follows a sequential multi-stage architecture.

Rule-based Extraction

The pipeline includes curated patterns to detect instructions such as "start", "stop", "resume", "hold", or "continue", including common clinical shorthand terms such as “d/c”, “dc”, “taper off”, “increase to”, and others.

Biomedical NER

A pretrained model (d4data/biomedical-ner-all) performs entity recognition in batch mode. Only drug-related labels are retained for further processing.

Drug Dictionary

A text file containing drug names is loaded at startup. Each name is normalized to improve pattern matching. The dictionary is used to identify medications that may not appear as NER entities.

Candidate Construction

Outputs from the three components (rule-based, NER, dictionary) are merged. Each action is associated with the closest medication span in the text.

Confidence Assignment

The system assigns a confidence score based on NER model output and cross-source agreement.

LLM Review (Optional)

Low-confidence entries may be submitted to an LLM running locally through Ollama. The model is instructed to evaluate the clinical note comprehensively, reconstruct missing medications, and resolve ambiguous actions. Returned JSON is validated before saving.

Chunk Processing and Concurrency

To support large datasets, the pipeline processes notes in chunks. NER runs in batches, and LLM queries are parallelized by worker threads.

Installation

Clone the repository:

git clone https://github.com/<your-repo>/medication-reconciliation.git
cd medication-reconciliation


Create a virtual environment:

python -m venv venv


Activate the environment:

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate


Install requirements:

pip install -r requirements.txt

Installing Ollama (Optional for LLM Review)

Install Ollama from:

https://ollama.com/download

Once installed, pull a model suitable for clinical reasoning:

ollama pull llama3


Start the Ollama server:

ollama serve


If you do not enable LLM mode, the pipeline will run without Ollama.

Preparing the Drug Dictionary

Place a text file named drug_set.txt in the data/ directory.
Each line should contain a drug name in lowercase:

latanoprost
timolol
brimonidine
aspirin
hydrocodone


Ensure the file does not contain commas or additional formatting.

Running the Pipeline

The script accepts command-line arguments.

Basic execution:

python medication_extraction.py


This uses default values:

Input CSV: Sample_data_to_test.csv

Drug list: drug_set.txt

Output file: output_notes_df.csv

Chunk size: 100

LLM workers: 1

LLM enabled: True

Optional Arguments
python medication_extraction.py \
  --notes path/to/notes.csv \
  --drugs path/to/drug_set.txt \
  --output results.csv \
  --use_llm true \
  --chunk_size 200 \
  --llm_workers 2


Arguments not provided fallback to defaults defined in the script.

Logging

The script writes all logs to:

logs/med_recon_chunk_parallel.log


This includes:

Chunk progress updates

NER initialization

Number of detected actions

Records requiring LLM review

LLM JSON extraction failures

Any pipeline errors

Troubleshooting

If Ollama returns a 500 error:
Reduce the number of LLM workers:

--llm_workers 1


Restart the Ollama server:

ollama serve


If the NER model loads slowly:
Ensure a GPU is configured correctly or install CPU-only PyTorch.

If LLM JSON parsing fails:
The system falls back automatically and logs the raw output for inspection.

Extensibility

The pipeline is designed for easy modification:

Add more dictionary entries for broader coverage.

Replace the NER model without changing the pipeline structure.

Adjust chunk size for memory or performance considerations.

Extend LLM prompts for specialized domains.
