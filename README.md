# 📰 Entropy of Propaganda — Part 1: Multilingual Preprocessing Pipeline

> **Stage 01 / 03**
> Week 1: Literature Review, Data Acquisition & Modular Preprocessing

---

## 📌 Project context

| Part | Week | Focus |
|---|---|---|
| **01 — this stage** | 1 | Data acquisition + modular preprocessing pipeline |
| 02 | 2 | LLM deployment + cross-entropy → BPC calculation |
| 03 | 3 | Statistical analysis, plots, reporting |

Before entropy can be measured, the raw data has to be **one clean, consistent, deduplicated corpus**. That's what this stage delivers.

---

## 🧠 What it does

`preprocessing_pipeline.py` walks a folder of raw source files — Narrative/Hierarchy datasets, Weak-Label and HQP datasets, plus assorted tweet/news/Telegram exports — and outputs one cleaned corpus:

```
clean_propaganda_dataset.jsonl
clean_propaganda_dataset.parquet
```

Input spans **5 file formats** (`.parquet`, `.csv`, `.tsv`, `.jsonl`, `.txt`) and **3 text domains** (tweets, news articles, Telegram posts) — the pipeline normalizes all of them through one shared cleaning path.


## ⚙️ Pipeline stages

```
Raw files (parquet/csv/tsv/jsonl/txt)
        │
        ▼
1. File discovery & priority ordering
        │
        ▼
2. Format-agnostic loading + text-column detection
        │
        ▼
3. Text normalization  (process())
        │
        ▼
4. Boilerplate removal  (static + statistical)
        │
        ▼
5. Quality filtering  (alpha ratio, length, language)
        │
        ▼
6. Global deduplication  (MD5 exact + MinHash/LSH near-dup)
        │
        ▼
7. Write to local disk → convert to Parquet → copy both to Drive
```

### 1. File discovery & prioritization
Recursively scans `drive/MyDrive/files_prop`, skips archives/system/mapping-meta files, and sorts by priority so higher-quality sources are ingested (and deduplicated against) first:

| Priority | Files matching |
|---|---|
| 1 | `narrative`, `hierarchy` |
| 2 | `labeled`, `hiqualprop`, `hqp` |
| 3 | everything else |

### 2. Format-agnostic loading
Each format gets its own loader, including a heuristic for `.txt` files that decides whether they're really delimited tabular data or free text (split by paragraph, or line as fallback). `find_text_column()` auto-detects the text column by common name, or by longest average string length among object columns.

### 3. Text normalization — `process()`
One function handles tweets, news scrapes, and Telegram exports in a single pass:
- Unwraps nested chat/JSON payloads (`extract_content()`) — e.g. LLM prompt logs where the real text sits inside a `Text:` field
- Strips URLs, `@mentions`, zero-width/invisible unicode, HTML entities & tags
- Splits `CamelCaseHashtags` into readable words
- Removes Telegram header artifacts (timestamps, `None`/media tokens) and Reuters/date-line leads
- Normalizes smart quotes/dashes → plain ASCII

### 4. Boilerplate removal — two tiers
| Tier | Method | Catches |
|---|---|---|
| Static | `BOILERPLATE_PATTERNS` regex list | Known UA/EN outlet CTAs, Reuters/Fox disclaimers, Telegram subscribe prompts |
| Statistical | `strip_generic_boilerplate_per_file()` | Short sentences that recur across many docs *within a file* **and** cluster near the start/end — the signature of an unlisted header/footer |

The statistical layer lets the pipeline generalize to sources nobody hand-wrote a regex for.

### 5. Quality filtering
- Alphabetic-character ratio > 0.5 (drops numeric/symbol noise)
- Cleaned text length ≥ 100 chars
- `langdetect` → keep only `uk` and `en`

### 6. Global deduplication
| Layer | Method | Parameters |
|---|---|---|
| Exact | MD5 hash of cleaned text | — |
| Near-duplicate | MinHash + LSH over 3-gram shingles | `num_perm=128`, `threshold=0.90` |

Applied **globally across all files**, not per-file — a duplicate in one dataset suppresses the same text reappearing in another. Combined with priority ordering, higher-quality sources win when a near-duplicate shows up later.

### 7. Local-first write, then sync to Drive
Rows are written to JSONL **on local disk** (`/content/...`) as each file finishes, with a `tqdm` progress bar over `files_to_process`. Writing locally instead of straight to a mounted Drive avoids the I/O overhead/latency of Drive's virtual filesystem while processing thousands of small writes.

Each file is still wrapped in its own `try/except`, but failures are now **logged** (`print(f"Error processing file {file_path.name}: {e}")`) rather than silently swallowed — easier to debug which sources are malformed.

Once processing finishes, the local JSONL is converted to Parquet via DuckDB, and **both** the JSONL and Parquet are copied to Google Drive (`shutil.copy`) as the final step — so Drive only sees fully-finished files, not thousands of incremental writes.

---

## 🤔 Why it's built this way

- **One script, many formats** — three genuinely different text domains in five file formats; a shared `process()` core keeps cleaning logic auditable in one place instead of three parallel pipelines.
- **Two-tier boilerplate removal** — static patterns alone don't scale to unseen sources; a purely statistical pass alone is too aggressive on small files. Together they cover both known and unknown noise.
- **Global, incremental dedup** — propaganda datasets are heavily syndicated (same article/post scraped many times). Deduplicating *before* entropy calculation is essential — repeated text would otherwise skew the BPC distributions analyzed in Week 3.
- **Priority ordering** — ensures better-annotated sources (narrative/hierarchy, HQP) are kept over lower-quality duplicates found later.
- **BPC needs clean input** — leftover HTML, boilerplate, or non-target-language text would inflate/distort entropy for reasons unrelated to the actual rhetorical signal being studied. This stage removes those confounds before Week 2's models ever see the data.

---

## 📤 Output

A single deduplicated, Ukrainian/English-only corpus with `source_file` provenance preserved per row:

```jsonc
{
  "text_col": "...",
  "processed": "cleaned text...",
  "alpha_ratio": "0.87",
  "detected_lang": "uk",
  "source_file": "drive/MyDrive/files_prop/hqp_labeled.parquet"
}
```


