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
├── process/                   # Processing scripts (one per dataset + tagger)
│   ├── qmsum.py               # 1. QMSum meeting transcripts
│   ├── fortress.py             # 2. Fortress adversarial/benign prompts
│   ├── confaide.py             # 3. ConfAIde privacy norms
│   ├── agentleak.py            # 4. AgentLeak PII scenarios
│   ├── sensitive_doc.py        # 5. Sensitive document classification
│   ├── confidential_biz.py     # 6. Confidential business excerpts
│   ├── pku_saferlhf.py         # 7. PKU-SafeRLHF national security
│   ├── govreport.py            # 8. GovReport summaries
│   ├── us_bizdata.py           # 9. US business data
│   └── gpt_tagger.py           # GPT-4 inference (3-perspective tagging)
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

### 1. Run with sampling (default: 10 per dataset)

```bash
cd scripts
bash run.sh
```

### 2. Run with custom sample size

```bash
bash run.sh 50       # 50 samples per dataset
```

### 3. Run full data (no sampling)

```bash
bash run.sh all
```

### 4. GPT Tagging

Edit `scripts/run.sh` and set your config at the bottom:

```bash
API_KEY="sk-xxx"           # your OpenAI API key
MAX_SAMPLES_TAG=10         # samples to tag per file (remove for all)
MODEL="gpt-4"              # model to use
```

The tagger runs automatically after data processing. It reads from `processed_data/` and writes to `tagged_data/`.

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

No manual data setup is required for most datasets.

## Requirements

```
openai
tiktoken
pandas
pyarrow
```