# Privacy_Benchmark

A multi-dataset privacy and sensitivity tagging benchmark. This project collects 9 diverse datasets, normalizes them into a unified JSONL format (`{"prompt": ...}`), and uses GPT-4 to generate sensitivity tags from three perspectives: **individual**, **company**, and **government**.

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
│   └── gpt_tagger.py           # GPT-4 tagging (3-perspective)
│
├── scripts/                   # Entry point
│   └── run.sh                  # Main runner script
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
| 5 | Sensitive Doc | [mouhamet/sensitive_document_classification](https://huggingface.co/datasets/mouhamet/sensitive_document_classification) | full document text (one .txt = one prompt) |
| 6 | Confidential Biz | [Rohit-D/synthetic-confidential-...](https://huggingface.co/datasets/Rohit-D/synthetic-confidential-information-injected-business-excerpts) | Excerpt field |
| 7 | PKU-SafeRLHF | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | prompt + national security response |
| 8 | GovReport | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) | summary field |
| 9 | US BizData | [ttn1410/US_BussinessData](https://huggingface.co/datasets/ttn1410/US_BussinessData) | description field |

## Quick Start

```bash
cd scripts
sh run.sh
```

All configuration is done by editing `scripts/run.sh` directly.

### Sampling Config

At the top of `run.sh`, find:

```bash
NUM_SAMPLES=10                 # <-- set number per dataset, or "all" for full data
```

- Sample 10 per dataset: `NUM_SAMPLES=10`
- Sample 50: `NUM_SAMPLES=50`
- Full data: `NUM_SAMPLES="all"`

### GPT Tagging Config

Also at the top of `run.sh`:

```bash
API_KEY=""                     # <-- paste your OpenAI API key here
MODEL="gpt-4o"                 # <-- model to use
```

The tagger reuses `NUM_SAMPLES` — if you set `NUM_SAMPLES=10`, GPT tags 10 samples per file; if `NUM_SAMPLES="all"`, it tags everything.

The tagger runs automatically after data processing. It reads from `processed_data/` and writes to `tagged_data/`.

> **Note:** If `processed_data/` already contains JSONL files, the data processing step (1-9) is skipped automatically. To re-process, delete the `processed_data/` directory first.

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
    {"tag": "product strategy", "perspective": "company", "explanation": "Discusses unreleased product design decisions."},
    {"tag": "employee opinion", "perspective": "individual", "explanation": "Contains identifiable personal views from team members."},
    {"tag": "committee deliberation", "perspective": "government", "explanation": "Involves policy review discussion in a parliamentary setting."}
  ]
}
```

## Auto-Download

Most processing scripts will automatically download their dataset if the local path is missing:
- GitHub repos → `git clone`
- HuggingFace datasets → download parquet/zip via URL
