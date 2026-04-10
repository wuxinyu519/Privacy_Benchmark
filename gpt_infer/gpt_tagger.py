#!/usr/bin/env python3
"""
This script processes instruction datasets, generates structured sensitivity tags
from three perspectives (individual, company, government) using GPT-4,
and saves the results in a new file.
"""

import os
import json
import sys
from pathlib import Path
from openai import OpenAI
import time
import tiktoken


SYSTEM_PROMPT = """You are a privacy and sensitivity tagging system. You analyze text and produce sensitivity tags from three perspectives: individual, company, and government.

Rules:
- Output ONLY valid JSON, no markdown, no explanation outside JSON.
- Each tag is an object with "tag", "perspective", and "explanation".
- "perspective" must be one of: "individual", "company", "government".
- Produce 2-5 tags per perspective if relevant. If a perspective is not applicable, skip it.
- Do NOT reveal, repeat, or restate any sensitive information (PII, secrets, etc.) in your tags or explanations.
- Tag names should be concise (1-4 words). Explanations should be 1 sentence.

Output format:
[{"tag": str, "perspective": str, "explanation": str}, ...]"""


# ============================================================
# Per-dataset prompt templates
# ============================================================
TEMPLATES = {

# 1. QMSum - Meeting transcripts (Product + Committee)
"qmsum": """This text is from the QMSum dataset — a collection of meeting transcripts from product design meetings and parliamentary committee sessions. Participants discuss product features, design decisions, market strategies, policy issues, and organizational matters.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 2a. Fortress Benign - Safe prompts
"fortress_benign": """This text is from the Fortress dataset (benign split) — a collection of user requests that appear safe and non-malicious. These are everyday questions or tasks a user might ask an AI assistant.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 2b. Fortress Risky - Adversarial prompts
"fortress_risky": """This text is from the Fortress dataset (adversarial split) — a collection of user prompts designed to test AI safety. These prompts may use social engineering, obfuscation, roleplay, or other techniques to attempt to elicit harmful, dangerous, or illegal information.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 3. ConfAIde - Privacy norms and contextual integrity
"confaide": """This text is from the ConfAIde dataset — a benchmark for evaluating privacy norms and contextual integrity. It contains scenarios about information sharing between people in various relationships (co-workers, classmates, family members) and asks whether sharing certain private information is appropriate given the context.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 4. AgentLeak - Agent scenarios with PII in private vault
"agentleak": """This text is from the AgentLeak dataset — a benchmark for testing whether AI agents leak private information. It contains a user query paired with private records (patient data, financial records, legal cases, employee files, etc.) that the agent has access to. The records may contain PII such as SSNs, medical diagnoses, financial details, and more.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Do NOT repeat any actual PII values in your tags.
Output JSON array of tags:""",

# 5. Sensitive Document Classification - Business documents/emails
"sensitive_doc": """This text is from the Sensitive Document Classification dataset — a collection of synthetic business documents including memos, emails, reports, policy documents, meeting minutes, and internal communications across various business domains and sensitivity levels.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 6. Confidential Business Excerpts - Synthetic business text with injected secrets
"confidential_biz": """This text is from a dataset of synthetic business excerpts that contain injected confidential information. Each excerpt describes a business scenario where certain sentences reveal information that should not be publicly disclosed — such as unreleased plans, internal challenges, or strategic decisions.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 7. PKU-SafeRLHF - National security harmful Q&A
"pku_national_security": """This text is from the PKU-SafeRLHF dataset, specifically filtered for content flagged as "Endangering National Security." It contains a user query paired with an AI-generated response that was labeled as harmful. The content may involve topics related to national security, public safety, or other high-risk areas.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Do NOT repeat any harmful instructions in your tags.
Output JSON array of tags:""",

# 8. GovReport - Government report summaries
"govreport": """This text is from the GovReport dataset — a collection of summaries of U.S. government reports from the Government Accountability Office (GAO) and Congressional Research Service (CRS). These summaries cover topics such as defense, healthcare, federal programs, budgets, audits, and policy evaluations.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

# 9. US Business Data - Company descriptions
"us_bizdata": """This text is from a dataset of US publicly traded company descriptions. Each entry describes a company's business operations, products, services, market segments, and strategic focus areas. The data includes companies across sectors such as technology, finance, healthcare, and more.

Text:
{instruction}

Analyze this text and generate sensitivity tags from three perspectives: individual, company, and government.
Output JSON array of tags:""",

}


class DatasetTagger:

    def __init__(self, api_key: str, model: str = "gpt-4", output_dir: str = "tagged_data"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.output_dir = output_dir
        self.encoding = tiktoken.encoding_for_model(model)
        os.makedirs(output_dir, exist_ok=True)

    def truncate_text(self, text: str, max_tokens: int = 600) -> str:
        """Keep first 300 and last 300 tokens."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        half = max_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
        return self.encoding.decode(truncated_tokens)

    def get_dataset_type(self, filename: str) -> str:
        """Determine template type based on filename keywords."""
        filename_lower = filename.lower()

        mapping = [
            (['qmsum'], 'qmsum'),
            (['fortress_benign'], 'fortress_benign'),
            (['fortress_risky'], 'fortress_risky'),
            (['confaide'], 'confaide'),
            (['agentleak'], 'agentleak'),
            (['sensitive_doc'], 'sensitive_doc'),
            (['confidential_biz'], 'confidential_biz'),
            (['pku_national_security'], 'pku_national_security'),
            (['govreport'], 'govreport'),
            (['us_bizdata'], 'us_bizdata'),
        ]

        for keywords, dtype in mapping:
            if any(k in filename_lower for k in keywords):
                return dtype

        return "sensitive_doc"  # safe default

    def get_prompt_template(self, dataset_type: str) -> str:
        """Get prompt template for dataset type."""
        return TEMPLATES.get(dataset_type, TEMPLATES["sensitive_doc"])

    def generate_tags(self, instruction: str, dataset_type: str, max_retries: int = 3):
        """Generate 3-perspective sensitivity tags via GPT. Retry up to max_retries if 0 tags returned."""
        truncated_instruction = self.truncate_text(instruction)
        prompt = self.get_prompt_template(dataset_type).format(instruction=truncated_instruction)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                content = response.choices[0].message.content.strip()

                # Strip markdown fences if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                tags = json.loads(content)

                if isinstance(tags, list) and len(tags) > 0:
                    return tags

                # 0 tags, retry
                if attempt < max_retries:
                    print(f"[retry {attempt}/{max_retries}: 0 tags]", end=' ')
                    time.sleep(1)

            except Exception as e:
                if attempt < max_retries:
                    print(f"[retry {attempt}/{max_retries}: {e}]", end=' ')
                    time.sleep(1)
                else:
                    print(f"  Error after {max_retries} attempts: {e}")

        return []

    def process_file(self, filepath: str, max_samples: int = None):
        """Process a single JSONL file."""
        filename = os.path.basename(filepath)
        dataset_type = self.get_dataset_type(filename)

        print(f"\n{'='*60}")
        print(f"File: {filename} | Type: {dataset_type}")
        print(f"{'='*60}")

        # Read data
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue

        print(f"Total #: {len(data)}")

        # Limit samples
        if max_samples and max_samples < len(data):
            data = data[:max_samples]
            print(f"Maximum processing: {max_samples}")

        output_path = os.path.join(
            self.output_dir,
            filename.replace('.jsonl', '_tagged.jsonl')
        )

        # Check already processed (for resume)
        processed_count = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for _ in f)
            print(f"Already processed: {processed_count} samples, continuing...")

        # Process in batches
        batch = []
        for idx, item in enumerate(data[processed_count:], processed_count + 1):
            instruction = item.get('prompt', '')
            if not instruction:
                continue

            print(f"[{idx}/{len(data)}]", end=' ')

            tags = self.generate_tags(instruction, dataset_type)

            batch.append({
                "prompt": instruction,
                "ground_truth": tags
            })

            # Count tags per perspective
            perspectives = {}
            for t in tags:
                p = t.get('perspective', 'unknown')
                perspectives[p] = perspectives.get(p, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(perspectives.items()))
            print(f"{len(tags)} tags ({summary})")

            # Save every 10 records
            if len(batch) >= 10:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for result in batch:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"  Saved {idx} records.")
                batch = []

            time.sleep(0.5)

        # Save remaining
        if batch:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in batch:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\nSaved: {output_path}")

    def process_directory(self, input_dir: str, max_samples: int = None):
        """Process all JSONL files in a directory."""
        jsonl_files = sorted(Path(input_dir).glob('*.jsonl'))

        print(f"\nFound {len(jsonl_files)} files")

        for filepath in jsonl_files:
            self.process_file(str(filepath), max_samples)

        print(f"\n{'='*60}")
        print(f"Done, output: {self.output_dir}")
        print(f"{'='*60}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python gpt_tagger.py <input_dir> <api_key> [max_samples] [output_dir] [model]")
        sys.exit(1)

    input_dir = sys.argv[1]
    api_key = sys.argv[2]
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "tagged_data"
    model = sys.argv[5] if len(sys.argv) > 5 else "gpt-4"

    tagger = DatasetTagger(api_key, model, output_dir)
    tagger.process_directory(input_dir, max_samples)


if __name__ == "__main__":
    main()