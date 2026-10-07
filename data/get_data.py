import os

# 公司網路會攔截 HTTPS：使用 Windows 憑證庫，並關閉 hf-xet
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import truststore

truststore.inject_into_ssl()

from datasets import load_dataset
import json
from pathlib import Path

ds = load_dataset("glaiveai/glaive-function-calling-v2")

out_dir = Path(__file__).parent / "dataset"
out_dir.mkdir(exist_ok=True)
for split, data in ds.items():
    out_path = out_dir / f"glaive_function_calling_v2_{split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{split}: {len(data)} 筆 → {out_path}")
