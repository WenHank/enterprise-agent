"""
Convert glaive-function-calling-v2 to Llama-Factory sharegpt format and split into train / eval / test.

Outputs:
  train.jsonl / eval.jsonl / test.jsonl
  dataset_info.json  (dataset registry read by Llama-Factory)

Example sharegpt sample:
{
  "conversations": [
    {"from": "human",         "value": "I need to generate an invoice for John Doe..."},
    {"from": "function_call", "value": "{\"name\": \"generate_invoice\", \"arguments\": {...}}"},
    {"from": "observation",   "value": "{\"invoice_id\": \"INV12345\", ...}"},
    {"from": "gpt",           "value": "The invoice has been generated successfully..."}
  ],
  "system": "You are a helpful assistant.",
  "tools": "[{\"name\": \"generate_invoice\", \"description\": \"...\", \"parameters\": {...}}]"
}
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dataset"
SRC = DATA_DIR / "glaive_function_calling_v2_train.jsonl"
RATIOS = {"train": 0.8, "eval": 0.1, "test": 0.1}
SEED = 42

TOOLS_SYSTEM = "You are a helpful assistant with access to the following functions."
ROLE_MAP = {"USER": "human", "ASSISTANT": "gpt", "FUNCTION RESPONSE": "observation"}
TURN_RE = re.compile(r"(?:^|\n{2,})(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
QUOTED_ARGS_RE = re.compile(r"""^\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*'(.*)'\s*\}$""", re.S)


class SkipSample(Exception):
    pass


def parse_tools(system: str) -> list:
    """Extract each tool's JSON definition from the system prompt."""
    if TOOLS_SYSTEM not in system:
        return []
    text = system.split("Use them if required -", 1)[-1]
    decoder = json.JSONDecoder()
    tools, i = [], 0
    while True:
        i = text.find("{", i)
        if i == -1:
            return tools
        obj, i = decoder.raw_decode(text, i)
        tools.append(obj)


def parse_function_call(text: str) -> dict:
    """Parse the text after <functioncall> and turn arguments into a real JSON object."""
    text = text.strip()
    m = QUOTED_ARGS_RE.match(text)
    if m:  # common glaive form: "arguments": '{...}'
        return {"name": m.group(1), "arguments": json.loads(m.group(2))}
    call = json.loads(text)
    if isinstance(call.get("arguments"), str):
        call["arguments"] = json.loads(call["arguments"])
    return {"name": call["name"], "arguments": call.get("arguments", {})}


def convert(row: dict) -> dict:
    tools = parse_tools(row["system"])
    tool_names = {t["name"] for t in tools}

    parts = TURN_RE.split(row["chat"])
    if parts[0].strip():
        raise SkipSample("extra text before the first turn")

    conversations = []
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.replace("<|endoftext|>", "").strip()
        if role == "ASSISTANT" and "<functioncall>" in content:
            before, after = content.split("<functioncall>", 1)
            if before.strip():
                raise SkipSample("text before <functioncall>")
            call = parse_function_call(after)
            if call["name"] not in tool_names:
                raise SkipSample("calls an undefined tool")
            conversations.append({"from": "function_call", "value": json.dumps(call, ensure_ascii=False)})
        else:
            if not content:
                raise SkipSample("empty message")
            conversations.append({"from": ROLE_MAP[role], "value": content})

    # glaive often has the assistant say "Let me check..." and call the tool in the next turn.
    # Two assistant turns in a row are not allowed, so drop the filler and keep only the call.
    conversations = [
        turn for i, turn in enumerate(conversations)
        if not (turn["from"] == "gpt" and i + 1 < len(conversations)
                and conversations[i + 1]["from"] == "function_call")
    ]

    # Llama-Factory requires human/observation at odd positions and gpt/function_call at even positions
    for i, turn in enumerate(conversations):
        expected = ("human", "observation") if i % 2 == 0 else ("gpt", "function_call")
        if turn["from"] not in expected:
            raise SkipSample("invalid role order")
    if not conversations or len(conversations) % 2 == 1:
        raise SkipSample("does not end with an assistant turn")
    if conversations[0]["from"] != "human":
        raise SkipSample("does not start with a user turn")

    return {
        "conversations": conversations,
        "system": "You are a helpful assistant." if tools else "You are a helpful assistant, with no access to external functions.",
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
    }


def main():
    samples, skipped = [], Counter()
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                samples.append(convert(json.loads(line)))
            except SkipSample as e:
                skipped[str(e)] += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                skipped["JSON parse error"] += 1

    print(f"Converted {len(samples)} samples, skipped {sum(skipped.values())}")
    for reason, count in skipped.most_common():
        print(f"  - {reason}: {count}")

    random.Random(SEED).shuffle(samples)
    n = len(samples)
    n_train = int(n * RATIOS["train"])
    n_eval = int(n * RATIOS["eval"])
    splits = {
        "train": samples[:n_train],
        "eval": samples[n_train:n_train + n_eval],
        "test": samples[n_train + n_eval:],
    }

    dataset_info = {}
    for name, rows in splits.items():
        out_path = DATA_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} samples -> {out_path}")

        dataset_info[f"glaive_{name}"] = {
            "file_name": out_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        }

    info_path = DATA_DIR / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset_info -> {info_path}")


if __name__ == "__main__":
    main()
