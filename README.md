# Privacy_Benchmark

A multi-dataset privacy and sensitivity tagging benchmark. This project collects diverse datasets, normalizes them into a unified JSONL format (`{"prompt": ...}`), and uses GPT to generate sensitivity tags — grounded in the RedacBench policy list — from three perspectives: **individual**, **corporate**, and **government**.

## Project Structure

```
Privacy_Benchmark/
├── datasets/                  # Raw datasets (auto-downloaded if missing)
│   ├── QMSum/
│   ├── fortress_public/
│   ├── confaide/
│   ├── AgentLeak/
│   ├── sensitive_document_classification/
│   ├── confidential_biz/
│   ├── pku_saferlhf/
│   ├── govreport/
│   └── us_bizdata/
│
├── process/                   # Processing scripts (one per dataset)
│   ├── qmsum.py               # 1. QMSum meeting transcripts
│   ├── fortress.py             # 2. Fortress adversarial/benign prompts
│   ├── confaide.py             # 3. ConfAIde privacy norms
│   ├── agentleak.py            # 4. AgentLeak PII scenarios
│   ├── sensitive_doc.py        # 5. Sensitive document classification
│   ├── confidential_biz.py     # 6. Confidential business excerpts
│   ├── pku_saferlhf.py         # 7. PKU-SafeRLHF national security
│   ├── govreport.py            # 8. GovReport summaries
│   └── us_bizdata.py           # 9. US business data
│
├── gpt_infer/                 # GPT inference
│   └── gpt_tagger.py           # RedacBench-policy-grounded tagging (3-perspective)
│
├── redacBench/                 # Copy of RedacBench's policy list, used to ground tags
│   └── benchmark_dataset.json
│
├── scripts/                   # Entry points
│   ├── run_step1_process_data.sh   # Step 1: raw datasets -> processed_data/*.jsonl
│   └── run_step2_gpt_tagger.sh     # Step 2: processed_data/*.jsonl -> tagged_data/*.jsonl
│
├── processed_data/            # Output: normalized JSONL files (auto-created)
│
├── tagged_data/               # Output: GPT-tagged JSONL files (auto-created)
│
└── README.md
```

## Datasets

| # | Dataset | Source | What becomes `prompt` |
|---|---------|--------|----------------------|
| 1 | QMSum | [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) | query + meeting transcript context |
| 2 | Fortress | [ScaleAI/fortress_public](https://huggingface.co/datasets/ScaleAI/fortress_public) | benign_prompt / adversarial_prompt (2 files) |
| 3 | ConfAIde | [skywalker023/confaide](https://github.com/skywalker023/confaide) | tier1 info / tier2 scenario / tier3 query+context |
| 4 | AgentLeak | [Privatris/AgentLeak](https://github.com/Privatris/AgentLeak) | user_request + private vault records |
| 5 | Sensitive Doc *(disabled — source removed from HuggingFace)* | [mouhamet/sensitive_document_classification](https://huggingface.co/datasets/mouhamet/sensitive_document_classification) | full document text (one .txt = one prompt) |
| 6 | Confidential Biz | [Rohit-D/synthetic-confidential-...](https://huggingface.co/datasets/Rohit-D/synthetic-confidential-information-injected-business-excerpts) | Excerpt field |
| 7 | PKU-SafeRLHF | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | prompt + national security response |
| 8 | GovReport *(disabled — public data, low policy-violation yield)* | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) | summary field |
| 9 | US BizData *(disabled — public data, low policy-violation yield)* | [ttn1410/US_BussinessData](https://huggingface.co/datasets/ttn1410/US_BussinessData) | description field |

Datasets 5, 8, 9 are disabled by default in `run_step1_process_data.sh` and `gpt_infer/gpt_tagger.py` (commented out, not deleted — see comments in those files for why).

## Quick Start

Two independent steps — run data processing first, then tagging once you're
ready to spend API calls:

```bash
cd scripts
sh run_step1_process_data.sh    # raw datasets -> processed_data/*.jsonl
sh run_step2_gpt_tagger.sh      # processed_data/*.jsonl -> tagged_data/*.jsonl (calls GPT)
```

All configuration is done by editing these two scripts directly.

### Sampling Config

At the top of `run_step1_process_data.sh` and `run_step2_gpt_tagger.sh`, each has its own:

```bash
NUM_SAMPLES=10                 # <-- set number per dataset/file, or "all" for full data
```

They're independent on purpose — e.g. you can process 1000 samples per dataset once,
then tag them incrementally in smaller batches to control API cost. Both scripts
default to `NUM_SAMPLES=1` (cheap smoke test / prompt-quality check). For a real run,
set `NUM_SAMPLES=1000` in both.

When `NUM_SAMPLES` is less than the dataset size, `process/*.py` selects rows via
`random.sample()` with a fixed `random.seed(42)` — not the first N rows — so runs are
random but reproducible across re-runs.

**Row counts at `NUM_SAMPLES=1000`** (each dataset is capped by however many rows
actually exist in its source — several have fewer than 1000 available):

| Dataset | Rows at NUM_SAMPLES=1000 |
|---|---:|
| qmsum_product_committee | 1000 |
| agentleak | 1000 |
| confidential_biz | 1000 |
| pku_national_security | 1000 |
| fortress_benign | 500 |
| fortress_risky | 500 |
| confaide | 474 |
| **Total tagged rows** | **5474** |

`processed_data/*.jsonl` and `tagged_data/*_tagged.jsonl` are 1:1 — one row in,
one tagged row out. GovReport/US BizData/Sensitive Doc are disabled (see above), so
they contribute 0.

### GPT Tagging Config

At the top of `run_step2_gpt_tagger.sh`:

```bash
API_KEY=""                     # <-- paste your OpenAI API key here
MODEL="gpt-5.5"                # <-- model to use
POLICY_DATASET="../redacBench/benchmark_dataset.json"   # <-- RedacBench policy list used to ground tags
```

The tagger reads from `processed_data/` and writes to `tagged_data/`. Every tag it
generates is grounded in the RedacBench policy list (`redacBench/benchmark_dataset.json`,
copied from the `policy_aware_query_redaction` project) — see `gpt_infer/gpt_tagger.py`.

> **Note:** If `processed_data/` already contains JSONL files, `run_step1_process_data.sh`
> skips processing entirely. To re-process everything, delete `processed_data/` first;
> to backfill a single dataset, run its `process/<dataset>.py` script directly.

## Output Format

### Processed data (`processed_data/*.jsonl`)

```json
{"prompt": "Based on the following meeting transcript..."}
```

### Tagged data (`tagged_data/*_tagged.jsonl`)

```json
{
  "prompt": "Based on the following meeting transcript...",
  "ground_truth": [
    {
      "tag": "product strategy",
      "perspective": "corporate",
      "spans": ["we're delaying the launch until Q3"],
      "violated_policies": ["Discussion topics revealed"],
      "explanation": "Discusses unreleased product design decisions."
    },
    {
      "tag": "employee opinion",
      "perspective": "individual",
      "spans": ["I really don't trust the new manager"],
      "violated_policies": [],
      "explanation": "Contains identifiable personal views from team members."
    }
  ],
  "raw_response": "{\"tags\": [...]}",
  "error": ""
}
```

`violated_policies` entries are policy ids copied verbatim from `redacBench/benchmark_dataset.json`
(empty list if no specific policy applies but the tag is still plausibly sensitive).

`raw_response` is the model's raw text response for that sample (or its last attempt if all
retries failed), and `error` is the last exception/parse-failure message — empty string on
success. Both are always saved so a run with an empty `ground_truth` can be debugged directly
from the output file instead of needing to re-run.

## Auto-Download

Most processing scripts will automatically download their dataset if the local path is missing:
- GitHub repos → `git clone`
- HuggingFace datasets → download parquet/zip via URL
