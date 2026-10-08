"""Download the base model locally and run a 4-bit tool-calling smoke test.

Usage:
    python model/load_model.py               # download + test
    python model/load_model.py --skip-test   # download only

After downloading, set model_name_or_path in configs/*.yaml to the printed local path.
"""

import argparse
import json
import os
import re
from pathlib import Path

# The corporate network intercepts HTTPS: use the Windows cert store and disable hf-xet
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import truststore

truststore.inject_into_ssl()

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_DIR = Path(__file__).parent / "base" / MODEL_ID.split("/")[-1]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_meeting_room",
            "description": "預訂會議室",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string", "description": "會議室，例如 3F"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"},
                    "start": {"type": "string", "description": "開始時間，格式 HH:MM"},
                    "end": {"type": "string", "description": "結束時間，格式 HH:MM"},
                    "title": {"type": "string", "description": "會議名稱"},
                },
                "required": ["room", "date", "start", "end"],
            },
        },
    }
]

TEST_PROMPTS = [
    "今天是 2026-10-06。幫我訂明天下午兩點到三點的 3F 會議室，開專案週會",
    "你好，今天天氣如何？",  
]


def download() -> Path:
    path = snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    print(f"模型已下載到：{path}")
    return Path(path)


def smoke_test(model_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA，請確認安裝的是 cu128 版 torch")
    print(f"GPU：{torch.cuda.get_device_name(0)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    print(f"VRAM 使用：{torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    for prompt in TEST_PROMPTS:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tools=SAMPLE_TOOLS,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        reply = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        print(f"\nUser：{prompt}\nModel：{reply.strip()}")
        for call in re.findall(r"<tool_call>(.*?)</tool_call>", reply, re.S):
            try:
                print(f"→ tool_call 解析成功：{json.loads(call)}")
            except json.JSONDecodeError:
                print(f"→ tool_call JSON 格式錯誤：{call.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-test", action="store_true", help="只下載，不載入測試")
    args = parser.parse_args()

    model_path = download()
    if not args.skip_test:
        smoke_test(model_path)


if __name__ == "__main__":
    main()
