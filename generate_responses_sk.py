#!/usr/bin/env python3
"""Generate model responses for prompts in a JSONL file via OpenAI-compatible API.

Reads prompts from a JSONL file (e.g. data/sk_input_data.jsonl), sends each
prompt to the model, and writes {"prompt": ..., "response": ...} lines to the
output file.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 generate_responses_sk.py -i data/sk_input_data.jsonl -o out_responses.jsonl

    # Custom endpoint / local OpenAI-compatible server:
    python3 generate_responses_sk.py -i data/sk_input_data.jsonl -o out_responses.jsonl \\
        --base-url http://localhost:8000/v1 --api-key mykey --model mistral-7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def get_response(client, prompt: str, model: str, retries: int = 3) -> str | None:
    """Send a prompt to the model and return the response text."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[error] API error after {retries} attempts: {exc}", file=sys.stderr)
                return None
            wait = 2 ** attempt
            print(f"[warn] API error (attempt {attempt + 1}/{retries}): {exc}. Retrying in {wait}s…", file=sys.stderr)
            time.sleep(wait)
    return None


def build_client(api_key: str, base_url: str | None):
    from openai import OpenAI
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model responses for prompts in a JSONL file.")
    parser.add_argument("-i", "--input", default="data/sk_input_data.jsonl", help="Input JSONL file path (default: data/sk_input_data.jsonl)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. Falls back to $OPENAI_API_KEY if not provided.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL for OpenAI-compatible API endpoints (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--retries", type=int, default=3, help="Retry count on transient errors (default: 3)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = build_client(api_key, args.base_url)

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[error] Line {lineno}: invalid JSON – {exc}", file=sys.stderr)
                sys.exit(1)

            prompt = obj.get("prompt", "")
            response = get_response(client, prompt, args.model, args.retries)
            out = {"prompt": prompt, "response": response}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[info] Processed line {lineno}", file=sys.stderr)

    print(f"Done. Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
