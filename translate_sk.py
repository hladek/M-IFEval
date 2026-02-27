#!/usr/bin/env python3
"""Translate all string values in a JSONL file from English into Slovak via OpenAI.

Key names are preserved. Non-string values (numbers, booleans, null) are passed through unchanged.
Nested dicts and lists are traversed recursively.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 translate_jsonl_to_sk.py -i input.jsonl -o output.jsonl

    # Custom endpoint / local OpenAI-compatible server:
    python3 translate_jsonl_to_sk.py -i input.jsonl -o output.jsonl \\
        --base-url http://localhost:8000/v1 --api-key mykey --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

SYSTEM_PROMPT = (
    "You are a translation engine. "
    "You will receive a JSON object. "
    "Translate all string values from English into Slovak. "
    "Keep all JSON keys unchanged. "
    "Do not translate keys, only values. "
    "Preserve the original JSON structure exactly, including nested objects and arrays. "
    "Return only valid JSON with no extra explanation."
)


def translate_object(client, obj: Any, model: str, retries: int = 3) -> Any:
    """Send a whole JSON object to the API and return the translated object."""
    text = json.dumps(obj, ensure_ascii=False)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            return json.loads(result)
        except json.JSONDecodeError as exc:
            if attempt == retries - 1:
                raise ValueError(f"Model returned invalid JSON: {exc}") from exc
            wait = 2 ** attempt
            print(f"[warn] Invalid JSON in response (attempt {attempt + 1}/{retries}): {exc}. Retrying in {wait}s…", file=sys.stderr)
            time.sleep(wait)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[warn] API error (attempt {attempt + 1}/{retries}): {exc}. Retrying in {wait}s…", file=sys.stderr)
            time.sleep(wait)
    return obj  # unreachable


def build_client(api_key: str, base_url: str | None):
    from openai import OpenAI  # imported here so error message is clear
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate JSONL string values EN→SK via OpenAI.")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
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

            translated = translate_object(client, obj, args.model, args.retries)
            fout.write(json.dumps(translated, ensure_ascii=False) + "\n")
            print(f"[info] Translated line {lineno}", file=sys.stderr)

    print(f"Done. Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
