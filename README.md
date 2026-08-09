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

---

## ⚙️ Pipeline stages

```
Raw files (parquet/csv/tsv/jsonl/txt)
        │
        ▼
1. Initial class split (propaganda vs. baseline, 1:2 sample cap)
        │
        ▼
2. File discovery & priority ordering
        │
        ▼
3. Format-agnostic loading + text-column detection
        │
        ▼
4. Text normalization  (process())
        │
        ▼
5. Boilerplate removal  (static + structural + statistical)
        │
        ▼
6. Quality filtering  (alpha ratio, length, language)
        │
        ▼
7. Global deduplication  (MD5 exact + MinHash/LSH near-dup)
        │
        ▼
8. Write to local disk → convert to Parquet → copy both to Drive
```

### 1. Initial class split

The source `messages.csv` is split by `binary_label` into `messages_propaganda.csv` and `baseline_messages.csv`. The baseline is capped at **up to 2× the propaganda count** (`random_state=42`), not a strict 1:1 — cleaning, quality filtering, and deduplication downstream remove an uneven share of rows from each class (propaganda text tends to be more heavily syndicated/duplicated), so an exact 1:1 split here wouldn't survive to the final corpus anyway.

> ⚠️ **Known follow-up**: this initial 1:2 cap is a coarse pre-filter, not the final balance. Before entropy is computed in Part 2/3, both cleaned classes should be re-equalized to the same final row count (`n = min(len(prop_clean), len(base_clean))`), sampled *after* all filtering/dedup — not before. Comparing entropy across unevenly sized samples confounds sample size with the linguistic signal being studied.

### 2. File discovery & prioritization

Recursively scans `drive/MyDrive/files_prop`, skips archives/system/mapping-meta files, and sorts by priority so higher-quality sources are ingested (and deduplicated against) first:

| Priority | Files matching |
|---|---|
| 1 | `narrative`, `hierarchy` |
| 2 | `labeled`, `hiqualprop`, `hqp` |
| 3 | everything else |

### 3. Format-agnostic loading

Each format gets its own loader, including a heuristic for `.txt` files that decides whether they're really delimited tabular data or free text (split by paragraph, or line as fallback). `find_text_column()` auto-detects the text column by common name, or by longest average string length among object columns.

### 4. Text normalization — `process()`

One function handles tweets, news scrapes, and Telegram exports in a single pass:
- Unwraps nested chat/JSON payloads (`extract_content()`) — e.g. LLM prompt logs where the real text sits inside a `Text:` field
- Strips URLs, `@mentions`, zero-width/invisible unicode, HTML entities & tags
- Splits `CamelCaseHashtags` into readable words
- Removes Telegram header artifacts (timestamps, `None`/media tokens) and Reuters/date-line leads
- Strips embedded ALL-CAPS headline insertions (a Fox News-style artifact where unrelated article titles are spliced mid-paragraph), detected structurally rather than by literal pattern
- Normalizes smart quotes/dashes → plain ASCII
- Caps document length (`MAX_DOC_CHARS`) after cleaning, so rare multi-thousand-word longform outliers don't disproportionately skew downstream entropy stats relative to the typical short post/tweet

### 5. Boilerplate removal — three layers

| Layer | Method | Catches |
|---|---|---|
| Static | `BOILERPLATE_PATTERNS` regex list | Known UA/EN outlet CTAs, Reuters/Fox disclaimers, Telegram subscribe prompts, "reminder/нагадаємо"-style structural markers |
| Structural | ALL-CAPS run detector | Embedded unrelated headlines with no fixed wording, so no single regex could ever cover them |
| Statistical | `strip_generic_boilerplate_per_file()` (cross-doc) + `strip_intradoc_repeats()` (within-doc) | Short sentences that recur across many docs in a file **and** cluster near start/end (unlisted header/footer), plus sentences repeated 3+ times inside one long document |

The statistical layers let the pipeline generalize to sources nobody hand-wrote a regex for, while staying conservative enough not to strip intentional rhetorical repetition (slogans, refrains) that the entropy analysis is actually meant to measure — a pure token-window dedup was tried and dropped for exactly this reason.

### 6. Quality filtering

- Alphabetic-character ratio > 0.5 (drops numeric/symbol noise)
- Cleaned text length ≥ 100 chars
- `langdetect` → keep only `uk` and `en`

### 7. Global deduplication

| Layer | Method | Parameters |
|---|---|---|
| Exact | MD5 hash of cleaned text | — |
| Near-duplicate | MinHash + LSH over 3-gram shingles | `num_perm=128`, `threshold=0.90` |

Applied **globally across all files**, not per-file — a duplicate in one dataset suppresses the same text reappearing in another. Combined with priority ordering, higher-quality sources win when a near-duplicate shows up later.

### 8. Local-first write, then sync to Drive

Rows are written to JSONL **on local disk** (`/content/...`) as each file finishes, with a `tqdm` progress bar over `files_to_process`. Writing locally instead of straight to a mounted Drive avoids the I/O overhead/latency of Drive's virtual filesystem while processing thousands of small writes.

Each file is still wrapped in its own `try/except`, but failures are **logged** (`print(f"Error processing file {file_path.name}: {e}")`) rather than silently swallowed — easier to debug which sources are malformed.

Once processing finishes, the local JSONL is converted to Parquet via DuckDB, and **both** the JSONL and Parquet are copied to Google Drive (`shutil.copy`) as the final step — so Drive only sees fully-finished files, not thousands of incremental writes.

---

## 🤔 Why it's built this way

- **Loose initial split, exact final balance later** — cutting the baseline to an exact 1:1 before cleaning would just get thrown off again by uneven attrition through dedup/filtering; a looser 1:2 cap up front preserves enough headroom, with exact equalization deferred to right before entropy is computed.
- **One script, many formats** — three genuinely different text domains in five file formats; a shared `process()` core keeps cleaning logic auditable in one place instead of three parallel pipelines.
- **Layered boilerplate removal, kept conservative** — static + structural + statistical passes cover known, unpredictable, and per-file-unique noise respectively, but intentionally stop short of aggressive intra-document phrase dedup, since propaganda's repeated slogans/refrains are signal, not noise, for this study.
- **Global, incremental dedup** — propaganda datasets are heavily syndicated (same article/post scraped many times). Deduplicating *before* entropy calculation is essential — repeated text would otherwise skew the BPC distributions analyzed in Week 3.
- **Priority ordering** — ensures better-annotated sources (narrative/hierarchy, HQP) are kept over lower-quality duplicates found later.
- **Length cap** — a handful of multi-thousand-word longform articles can dominate an entropy distribution built mostly from short posts; capping avoids that without needing per-artifact regex chasing.
- **BPC needs clean input** — leftover HTML, boilerplate, or non-target-language text would inflate/distort entropy for reasons unrelated to the actual rhetorical signal being studied. This stage removes those confounds before Week 2's models ever see the data.

---

## 📤 Output

A deduplicated, roughly-balanced (final exact balance pending, see step 1), Ukrainian/English-only corpus with `source_file` provenance preserved per row:

```jsonc
{
  "text_col": "...",
  "processed": "cleaned text...",
  "alpha_ratio": "0.87",
  "detected_lang": "uk",
  "source_file": "drive/MyDrive/files_prop/hqp_labeled.parquet"
}
```
