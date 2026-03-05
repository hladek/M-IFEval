#!/usr/bin/env python3
"""Check that 'keyword', 'prompt_to_repeat', 'end_phrase' and 'forbidden_words' values in kwargs are present in the prompt."""

import json
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "data/sk_input_data.jsonl"

issues = []
total = 0

with open(input_file) as f:
    for line in f:
        record = json.loads(line)
        total += 1
        key = record["key"]
        prompt = record["prompt"]

        for i, kw in enumerate(record.get("kwargs", [])):
            instruction_id = record["instruction_id_list"][i]

            if "keyword" in kw:
                keyword = kw["keyword"]
                if keyword not in prompt:
                    issues.append(
                        f"key={key} [{instruction_id}]: keyword {keyword!r} not found in prompt"
                    )

            if "prompt_to_repeat" in kw:
                ptr = kw["prompt_to_repeat"]
                if ptr not in prompt:
                    issues.append(
                        f"key={key} [{instruction_id}]: prompt_to_repeat not found in prompt\n"
                        f"  prompt_to_repeat: {ptr!r}\n"
                        f"  prompt:           {prompt!r}"
                    )

            if "end_phrase" in kw:
                end_phrase = kw["end_phrase"]
                if end_phrase not in prompt:
                    issues.append(
                        f"key={key} [{instruction_id}]: end_phrase {end_phrase!r} not found in prompt"
                    )

            if "forbidden_words" in kw:
                for word in kw["forbidden_words"]:
                    if word not in prompt:
                        issues.append(
                            f"key={key} [{instruction_id}]: forbidden_word {word!r} not found in prompt"
                        )

print(f"Checked {total} records.")
if issues:
    print(f"Found {len(issues)} issue(s):\n")
    for issue in issues:
        print(issue)
else:
    print("All keyword, prompt_to_repeat, end_phrase and forbidden_words values are present in their prompts.")
