import os
os.environ["HF_TOKEN"] = "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"
from datasets import load_dataset
import json

print("TruthfulQA keys:")
ds1 = load_dataset("QCRI/AraDiCE-TruthfulQA", "TruthfulQA-msa", split="test")
print(list(ds1[0].keys()))
print("\nTruthfulQA[0]:")
print(json.dumps(ds1[0], indent=2, ensure_ascii=False)[:800])

print("\n\nHellaSwag keys:")
ds2 = load_dataset("QCRI/AraDiCE-HellaSwag", "HellaSwag-msa", split="validation")
print(list(ds2[0].keys()))
print("\nHellaSwag[0]:")
print(json.dumps(ds2[0], indent=2, ensure_ascii=False)[:800])
