# 📰 Entropy of Propaganda

This repository contains the code and methodology for measuring the linguistic predictability (entropy) of propaganda versus baseline text. 

## 📌 Project Context

The project is structured into three distinct stages.

| Stage | Focus |
| :--- | :--- |
| **01** | Data acquisition and modular preprocessing pipeline. |
| **02** | LLM deployment, cross-entropy extraction, and Bits Per Character (BPC) calculation. |
| **03** | Statistical analysis and visualizations (Note: Full analytical conclusions are detailed in the final external report). |

---

## ⚙️ Stage 01: Multilingual Preprocessing Pipeline

Before entropy can be measured, the raw data must be reduced to a clean, consistent, deduplicated corpus.

### What it does
`01_data_prep.ipynb` walks a folder of raw source files (Narrative/Hierarchy datasets, Weak-Label and HQP datasets, tweet/news/Telegram exports) and outputs a single cleaned corpus in both JSONL and Parquet formats. Input spans 5 file formats and 3 text domains, normalized through one shared cleaning path.

### Pipeline Stages

1. **Initial class split:** The source is split into propaganda and baseline (1:2 sample cap). This preserves headroom, as exact equalization is deferred until after deduplication.
2. **File discovery & priority ordering:** Scans the directory and sorts by priority so higher-quality sources (e.g., annotated datasets) are ingested and deduplicated against first.
3. **Format-agnostic loading:** Auto-detects text columns and handles `.parquet`, `.csv`, `.tsv`, `.jsonl`, and `.txt` seamlessly.
4. **Text normalization:** Unwraps JSON payloads, strips URLs/mentions/HTML, cleans Telegram/news artifacts, normalizes quotes, and enforces a strict length cap (`MAX_DOC_CHARS`) to prevent longform outliers from skewing entropy stats.
5. **Boilerplate removal:** Uses three layers (Static regex, Structural ALL-CAPS detectors, and Statistical intradoc/cross-doc repetition removal) to strip noise without destroying the rhetorical repetition inherent to propaganda.
6. **Quality filtering:** Enforces an alphabetic-character ratio > 0.5, a minimum length of 100 characters, and limits languages to `uk` and `en`.
7. **Global deduplication:** Applies exact MD5 hashing and MinHash+LSH (num_perm=128, threshold=0.90) globally across all files.
8. **Export:** Writes to local JSONL incrementally to avoid Drive I/O latency, converts to Parquet via DuckDB, and syncs the final files to Google Drive.

---

## 🧠 Stage 02: LLM Deployment & BPC Calculation

This stage takes the standardized dataset and processes it through causal language models to calculate cross-entropy, converting it into Bits Per Character (BPC).

### What it does
It scores the predictability of the text using next-token prediction. Implemented in two parallel environments:
`02_entropy_eval.py`: Primary pipeline for CUDA (Google Colab) using PyTorch, Hugging Face, and BitsAndBytes.

### Pipeline Stages

1. **Sentence Explosion:** Splits documents by terminal punctuation into discrete sentences. Only sentences between 120 and 200 characters are retained to maintain a consistent evaluation window.
2. **Model Loading:** Evaluates text across a suite of models (`gemma-3-1b-pt`, `Llama-3.2-3B`, `Qwen2.5-7B-Instruct`, etc.). Larger models utilize 4-bit NormalFloat (NF4) quantization to prevent OOM errors.
3. **Tokenization & Offset Mapping:** Tokenizes text with `return_offsets_mapping=True` to track character-to-token boundaries.
4. **Context Masking (70-Character Rule):** Masks the first 70 characters (loss set to 0 or -100) to account for the LLM "burn-in" phase where lack of prior context causes artificial loss spikes.
5. **Entropy to BPC Conversion:** Sums the cross-entropy loss across valid tokens (nats), divides by `np.log(2)` to convert to bits, and divides by the number of valid characters to derive the final standardized BPC.
6. **Export:** Outputs a dedicated `.csv` file for each evaluated model with calculated `total_bits`, `valid_chars`, and `bpc` columns.

---

## 📊 Stage 03: Statistical Analysis & Reporting

This final repository stage (`03_analysis.ipynb`) handles the aggregation and statistical testing of the generated BPC scores. 

**Note:** The code in this repository only generates the descriptive statistics, significance tests (e.g., Mann-Whitney U), and data visualizations (histograms, boxplots). **The full interpretation, deep-dive analysis, and final conclusions regarding the entropy of propaganda are exclusively documented in the project's formal written report.**
