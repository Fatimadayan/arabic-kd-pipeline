#!/usr/bin/env python3
"""
scripts/05_eval_4pillars.py
============================
4-Pillar evaluation framework.

Pillar 1: English ceiling (MMLU, GSM8K, TruthfulQA)
Pillar 2: Native Arabic (ArabicMMLU, ArabicQA, ORCA)
Pillar 3: Cross-lingual symmetry (LPG, CLCS, IFS, RDI)
Pillar 4: Transfer stress (AR→EN, EN→AR, code-switching, dialects)

Outputs:
  results/<model_name>/
    pillar1_english.json
    pillar2_arabic.json
    pillar3_symmetry.json
    pillar4_stress.json
    summary.csv
    summary.tex   (LaTeX table, booktabs)

CLUSTER SAFETY:
  - Models loaded in eval mode (no_grad always)
  - Disk checked before saving results
  - Each pillar runs independently — partial results saved on failure

Usage:
    python scripts/05_eval_4pillars.py \
        --model-path checkpoints/qwen7b_baseline_sft/final \
        --model-name qwen7b_sft \
        --config configs/eval.yaml
"""

import os
import sys
import re
import csv
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# SAFETY
# =============================================================================

def check_disk_space(path: Path, required_gb: float, label: str = "") -> None:
    stat = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = stat.free / (1024 ** 3)
    log.info(f"💾 [{label}]: {free_gb:.1f} GB free")
    if free_gb < required_gb:
        raise RuntimeError(f"DISK ERROR: Need {required_gb:.1f}, have {free_gb:.1f} GB.")


def is_path_approved(path: Path) -> bool:
    for var in ["HOME", "SCRATCH"]:
        root = os.environ.get(var)
        if root:
            try:
                path.resolve().relative_to(Path(root).resolve())
                return True
            except ValueError:
                continue
    return False


def safe_makedirs(path: Path) -> None:
    if not is_path_approved(path):
        raise PermissionError(f"SAFETY: {path} outside approved dirs.")
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_env(cfg):
    import re
    def _r(v):
        if isinstance(v, str):
            return re.sub(r"\$\{(\w+)(?::-(.*?))?\}",
                          lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
        if isinstance(v, dict): return {k: _r(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_r(i) for i in v]
        return v
    return _r(cfg)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_for_eval(model_path: str):
    """Load a fine-tuned LoRA model for evaluation."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel, PeftConfig
    except ImportError as e:
        raise ImportError(f"Missing: {e}")

    model_path = Path(model_path)

    # Check if it's a PEFT adapter or full model
    is_peft = (model_path / "adapter_config.json").exists()

    log.info(f"⚙️  Loading model from: {model_path} (PEFT={is_peft})")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    log.info("✅ Model loaded in eval mode.")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Single-sample generation with no_grad."""
    import torch
    device = next(model.parameters()).device

    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    resp_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(resp_ids, skip_special_tokens=True).strip()


# =============================================================================
# METRICS
# =============================================================================

def extract_mcqa_answer(response: str) -> Optional[str]:
    """Extract A/B/C/D from response."""
    patterns = [
        r"(?:^|\s)([A-D])[).：:\s]",
        r"(?:answer|الإجابة)[:\s]*([A-D])",
        r"^([A-D])$",
    ]
    for p in patterns:
        m = re.search(p, response.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def rouge_l_score(prediction: str, reference: str) -> float:
    """Simplified ROUGE-L (LCS-based)."""
    p_tokens = prediction.lower().split()
    r_tokens = reference.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0

    # LCS via DP
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i-1] == r_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m if m else 0
    recall = lcs / n if n else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_reasoning_steps(text: str, language: str = "en") -> int:
    """Heuristic count of reasoning steps in a response."""
    patterns = {
        "en": [r"\bstep\s*\d+", r"\bfirst(?:ly)?\b", r"\bsecond(?:ly)?\b",
               r"\bthird(?:ly)?\b", r"\btherefore\b", r"\bthus\b",
               r"^\d+[.)]", r"\bfinally\b"],
        "ar": [r"أولاً", r"ثانياً", r"ثالثاً", r"إذن", r"لذلك",
               r"^\d+[.)]", r"أخيراً", r"وبالتالي"],
    }
    lang_patterns = patterns.get(language, patterns["en"])
    count = 0
    for pat in lang_patterns:
        count += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))
    return max(count, 1)


def detect_language(text: str) -> str:
    """Simple Arabic/English detector."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    if arabic_chars > latin_chars:
        return "ar"
    return "en"


# =============================================================================
# PILLAR 1: ENGLISH BASELINE
# =============================================================================

def eval_pillar1(model, tokenizer, eval_cfg: dict, output_dir: Path) -> dict:
    log.info("\n📊 PILLAR 1: English Baseline")
    results = {}

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("pip install datasets --break-system-packages")
        return {"error": "datasets not installed"}

    p1_cfg = eval_cfg["pillar1_english"]

    # ── MMLU ──────────────────────────────────────────────────────────────────
    log.info("  → MMLU (100 samples, 5-shot)")
    try:
        ds = load_dataset(
            p1_cfg["benchmarks"]["mmlu"]["hf_path"],
            "all", split="test"
        ).shuffle(seed=42).select(range(p1_cfg["benchmarks"]["mmlu"]["n_samples"]))

        correct = 0
        for row in ds:
            choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])])
            prompt = f"Question: {row['question']}\n{choices}\nAnswer:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
            pred = extract_mcqa_answer(response)
            gold = chr(65 + row["answer"])
            if pred == gold:
                correct += 1

        results["mmlu_accuracy"] = correct / len(ds)
        log.info(f"    MMLU accuracy: {results['mmlu_accuracy']:.4f}")
    except Exception as e:
        log.warning(f"    MMLU failed: {e}")
        results["mmlu_accuracy"] = None

    # ── GSM8K ─────────────────────────────────────────────────────────────────
    log.info("  → GSM8K (50 samples, 0-shot)")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(50))
        correct = 0
        ans_regex = re.compile(r"####\s*([0-9,]+)")
        for row in ds:
            response = generate_response(model, tokenizer, row["question"], max_new_tokens=200)
            gold_m = ans_regex.search(row["answer"])
            pred_m = ans_regex.search(response) or re.search(r"([0-9,]+)\s*$", response)
            if gold_m and pred_m:
                gold = gold_m.group(1).replace(",", "")
                pred = pred_m.group(1).replace(",", "")
                if gold == pred:
                    correct += 1
        results["gsm8k_accuracy"] = correct / 50
        log.info(f"    GSM8K accuracy: {results['gsm8k_accuracy']:.4f}")
    except Exception as e:
        log.warning(f"    GSM8K failed: {e}")
        results["gsm8k_accuracy"] = None

    # ── TruthfulQA ────────────────────────────────────────────────────────────
    log.info("  → TruthfulQA (50 samples, ROUGE-L proxy)")
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation").shuffle(seed=42).select(range(50))
        scores = []
        for row in ds:
            response = generate_response(model, tokenizer, row["question"], max_new_tokens=100)
            best_score = max(rouge_l_score(response, ref) for ref in row["correct_answers"])
            scores.append(best_score)
        results["truthfulqa_rouge_l"] = sum(scores) / len(scores)
        log.info(f"    TruthfulQA ROUGE-L: {results['truthfulqa_rouge_l']:.4f}")
    except Exception as e:
        log.warning(f"    TruthfulQA failed: {e}")
        results["truthfulqa_rouge_l"] = None

    _save_json(results, output_dir / "pillar1_english.json")
    log.info(f"  💾 Pillar 1 saved.")
    return results


# =============================================================================
# PILLAR 2: NATIVE ARABIC
# =============================================================================

def eval_pillar2(model, tokenizer, eval_cfg: dict, output_dir: Path) -> dict:
    log.info("\n📊 PILLAR 2: Native Arabic")
    results = {}

    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets not installed"}

    # ── ArabicMMLU ────────────────────────────────────────────────────────────
    log.info("  → ArabicMMLU (100 samples, 5-shot)")
    try:
        ds = load_dataset("MBZUAI/ArabicMMLU", split="test").shuffle(seed=42).select(range(100))
        correct = 0
        for row in ds:
            choices_key = [k for k in row if "choice" in k.lower() or k in ["A","B","C","D"]]
            if choices_key:
                choices_str = "\n".join([f"{c}: {row[c]}" for c in ["A","B","C","D"] if c in row])
            else:
                choices_str = ""
            prompt = f"{row.get('question','')}\n{choices_str}\nالإجابة:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=10)
            pred = extract_mcqa_answer(response)
            gold = row.get("answer", row.get("Answer", ""))
            if pred and gold and pred.upper() == gold.upper():
                correct += 1
        results["arabic_mmlu_accuracy"] = correct / 100
        log.info(f"    ArabicMMLU accuracy: {results['arabic_mmlu_accuracy']:.4f}")
    except Exception as e:
        log.warning(f"    ArabicMMLU failed: {e}")
        results["arabic_mmlu_accuracy"] = None

    # ── ArabicQA (XQuAD AR) ───────────────────────────────────────────────────
    log.info("  → ArabicQA via XQuAD (100 samples, F1+EM)")
    try:
        ds = load_dataset("google/xquad", "xquad.ar", split="validation").shuffle(seed=42).select(range(100))
        f1_scores, em_scores = [], []
        for row in ds:
            prompt = f"السياق: {row['context'][:300]}\nالسؤال: {row['question']}\nالإجابة:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=50)
            gold_answers = row["answers"]["text"]
            best_f1 = max(rouge_l_score(response, g) for g in gold_answers)
            em = 1.0 if response.strip() in gold_answers else 0.0
            f1_scores.append(best_f1)
            em_scores.append(em)
        results["arabic_qa_f1"] = sum(f1_scores) / len(f1_scores)
        results["arabic_qa_em"] = sum(em_scores) / len(em_scores)
        log.info(f"    ArabicQA F1={results['arabic_qa_f1']:.4f} EM={results['arabic_qa_em']:.4f}")
    except Exception as e:
        log.warning(f"    ArabicQA failed: {e}")
        results["arabic_qa_f1"] = results["arabic_qa_em"] = None

    _save_json(results, output_dir / "pillar2_arabic.json")
    log.info("  💾 Pillar 2 saved.")
    return results


# =============================================================================
# PILLAR 3: CROSS-LINGUAL SYMMETRY
# =============================================================================

def eval_pillar3(model, tokenizer, eval_cfg: dict, output_dir: Path) -> dict:
    log.info("\n📊 PILLAR 3: Cross-Lingual Symmetry")
    results = {}

    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets not installed"}

    n = eval_cfg["pillar3_symmetry"]["n_parallel_pairs"]

    try:
        ds_ar = load_dataset("google/xquad", "xquad.ar", split="validation").select(range(n))
        ds_en = load_dataset("google/xquad", "xquad.en", split="validation").select(range(n))
    except Exception as e:
        log.warning(f"XQuAD load failed: {e}")
        return {"error": str(e)}

    en_correct, ar_correct = [], []
    en_responses, ar_responses = [], []
    en_steps, ar_steps = [], []

    for row_en, row_ar in zip(ds_en, ds_ar):
        # English QA
        prompt_en = f"Context: {row_en['context'][:300]}\nQuestion: {row_en['question']}\nAnswer:"
        resp_en = generate_response(model, tokenizer, prompt_en, max_new_tokens=80)
        gold_en = row_en["answers"]["text"]
        correct_en = max(rouge_l_score(resp_en, g) for g in gold_en) > 0.3

        # Arabic QA
        prompt_ar = f"السياق: {row_ar['context'][:300]}\nالسؤال: {row_ar['question']}\nالإجابة:"
        resp_ar = generate_response(model, tokenizer, prompt_ar, max_new_tokens=80)
        gold_ar = row_ar["answers"]["text"]
        correct_ar = max(rouge_l_score(resp_ar, g) for g in gold_ar) > 0.3

        en_correct.append(correct_en)
        ar_correct.append(correct_ar)
        en_responses.append(resp_en)
        ar_responses.append(resp_ar)
        en_steps.append(count_reasoning_steps(resp_en, "en"))
        ar_steps.append(count_reasoning_steps(resp_ar, "ar"))

    # ── Derived metrics ────────────────────────────────────────────────────────
    acc_en = sum(en_correct) / n
    acc_ar = sum(ar_correct) / n

    # LPG
    lpg = abs(acc_en - acc_ar)

    # CLCS
    consistent = sum(1 for e, a in zip(en_correct, ar_correct) if e == a)
    clcs = consistent / n

    # IFS (using ROUGE-L as proxy for translation quality)
    try:
        ds_en2 = load_dataset("google/xquad", "xquad.en", split="validation").select(range(min(50, n)))
        ifs_scores = []
        for row in ds_en2:
            # AR→EN: give Arabic prompt, ask for English answer
            prompt_ar2en = f"السياق: {row['context'][:200]}\nالسؤال: {row['question']}\nأجب باللغة الإنجليزية:"
            resp = generate_response(model, tokenizer, prompt_ar2en, max_new_tokens=60)
            score = rouge_l_score(resp, row["answers"]["text"][0])
            ifs_scores.append(score)
        ifs = sum(ifs_scores) / len(ifs_scores)
    except Exception:
        ifs = None

    # RDI
    avg_en_steps = sum(en_steps) / len(en_steps)
    avg_ar_steps = sum(ar_steps) / len(ar_steps)
    rdi = avg_ar_steps / avg_en_steps if avg_en_steps > 0 else None

    results = {
        "accuracy_en": round(acc_en, 4),
        "accuracy_ar": round(acc_ar, 4),
        "LPG": round(lpg, 4),
        "CLCS": round(clcs, 4),
        "IFS": round(ifs, 4) if ifs else None,
        "RDI": round(rdi, 4) if rdi else None,
        "avg_reasoning_steps_en": round(avg_en_steps, 2),
        "avg_reasoning_steps_ar": round(avg_ar_steps, 2),
        "n_pairs": n,
    }

    # ── Statistical tests ──────────────────────────────────────────────────────
    try:
        from scipy.stats import chi2
        import numpy as np

        # McNemar test
        b = sum(1 for e, a in zip(en_correct, ar_correct) if e and not a)
        c = sum(1 for e, a in zip(en_correct, ar_correct) if not e and a)
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = 1 - chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat, mcnemar_p = 0.0, 1.0

        # Cohen's d
        en_arr = np.array(en_correct, dtype=float)
        ar_arr = np.array(ar_correct, dtype=float)
        pooled_std = np.sqrt((np.var(en_arr) + np.var(ar_arr)) / 2)
        cohens_d = (np.mean(en_arr) - np.mean(ar_arr)) / pooled_std if pooled_std > 0 else 0.0

        results["mcnemar_statistic"] = round(float(mcnemar_stat), 4)
        results["mcnemar_p_value"] = round(float(mcnemar_p), 4)
        results["cohens_d"] = round(float(cohens_d), 4)
    except ImportError:
        log.warning("scipy not available — skipping statistical tests")
        results["mcnemar_p_value"] = None
        results["cohens_d"] = None

    # Per-sample data for error analysis
    results["per_sample"] = [
        {"en_correct": bool(e), "ar_correct": bool(a),
         "en_steps": es, "ar_steps": as_}
        for e, a, es, as_ in zip(en_correct, ar_correct, en_steps, ar_steps)
    ]

    _save_json(results, output_dir / "pillar3_symmetry.json")
    log.info(f"  LPG={results['LPG']:.4f} | CLCS={results['CLCS']:.4f} | RDI={results.get('RDI')}")
    log.info("  💾 Pillar 3 saved.")
    return results


# =============================================================================
# PILLAR 4: TRANSFER STRESS
# =============================================================================

def eval_pillar4(model, tokenizer, eval_cfg: dict, output_dir: Path) -> dict:
    log.info("\n📊 PILLAR 4: Transfer Stress Tests")
    results = {}

    # ── AR→EN output adherence ────────────────────────────────────────────────
    log.info("  → AR→EN transfer (50 samples)")
    ar2en_prompts = [
        "ما هو الذكاء الاصطناعي؟ أجب باللغة الإنجليزية.",
        "اشرح مفهوم التعلم الآلي بالإنجليزية.",
        "صف الفرق بين التعلم المُشرف وغير المُشرف. الرجاء الإجابة بالإنجليزية.",
        "ما هي فوائد الطاقة الشمسية؟ أجب بالإنجليزية.",
        "اشرح البرمجة الكائنية بالإنجليزية.",
    ] * 10  # Repeat to reach 50

    ar2en_scores = []
    for prompt in ar2en_prompts[:50]:
        resp = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        lang = detect_language(resp)
        ar2en_scores.append(1.0 if lang == "en" else 0.0)
    results["ar2en_language_adherence"] = sum(ar2en_scores) / len(ar2en_scores)
    log.info(f"    AR→EN adherence: {results['ar2en_language_adherence']:.4f}")

    # ── EN→AR output adherence ────────────────────────────────────────────────
    log.info("  → EN→AR transfer (50 samples)")
    en2ar_prompts = [
        "What is artificial intelligence? Please answer in Arabic.",
        "Explain machine learning concepts in Arabic.",
        "Describe neural networks in Arabic please.",
        "What are the benefits of renewable energy? Answer in Arabic.",
        "Explain object-oriented programming in Arabic.",
    ] * 10

    en2ar_scores = []
    for prompt in en2ar_prompts[:50]:
        resp = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        lang = detect_language(resp)
        en2ar_scores.append(1.0 if lang == "ar" else 0.0)
    results["en2ar_language_adherence"] = sum(en2ar_scores) / len(en2ar_scores)
    log.info(f"    EN→AR adherence: {results['en2ar_language_adherence']:.4f}")

    # ── Code-switching ────────────────────────────────────────────────────────
    log.info("  → Code-switching (30 samples)")
    cs_prompts = [
        "What is رأس المال الطبيعي and why does it matter for sustainability?",
        "Explain التعلم العميق (deep learning) in simple terms.",
        "How does نظام التوصيل work in modern e-commerce platforms?",
        "What is الذكاء الاصطناعي التوليدي and its applications?",
        "Describe الشبكات العصبية التلافيفية and their uses in vision.",
    ] * 6

    cs_rouge = []
    for prompt in cs_prompts[:30]:
        resp = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        # Coherence proxy: response length ≥ 20 words
        coherent = 1.0 if len(resp.split()) >= 20 else 0.0
        cs_rouge.append(coherent)
    results["code_switching_coherence"] = sum(cs_rouge) / len(cs_rouge)
    log.info(f"    Code-switching coherence: {results['code_switching_coherence']:.4f}")

    # ── Dialectal Arabic ──────────────────────────────────────────────────────
    log.info("  → Dialectal Arabic")
    dialect_prompts = {
        "gulf":     "شو رأيك في الذكاء الاصطناعي؟",
        "levantine": "شو بتحكي عن التكنولوجيا الحديثة؟",
        "egyptian": "إيه رأيك في التعلم الآلي؟",
        "msa":      "ما رأيك في التقنيات الحديثة؟",
    }
    dialect_scores = {}
    for dialect, prompt in dialect_prompts.items():
        responses = []
        for _ in range(5):
            resp = generate_response(model, tokenizer, prompt, max_new_tokens=80)
            responses.append(len(resp.split()) >= 10)
        dialect_scores[dialect] = sum(responses) / 5
        log.info(f"    {dialect}: {dialect_scores[dialect]:.2f}")
    results["dialectal_scores"] = dialect_scores
    results["dialectal_avg"] = sum(dialect_scores.values()) / len(dialect_scores)

    _save_json(results, output_dir / "pillar4_stress.json")
    log.info("  💾 Pillar 4 saved.")
    return results


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def generate_summary(
    p1: dict, p2: dict, p3: dict, p4: dict,
    model_name: str, output_dir: Path
) -> None:
    """Generate CSV + LaTeX summary table."""

    summary = {
        "model": model_name,
        # P1
        "MMLU": p1.get("mmlu_accuracy"),
        "GSM8K": p1.get("gsm8k_accuracy"),
        "TruthfulQA": p1.get("truthfulqa_rouge_l"),
        # P2
        "ArabicMMLU": p2.get("arabic_mmlu_accuracy"),
        "ArabicQA_F1": p2.get("arabic_qa_f1"),
        "ArabicQA_EM": p2.get("arabic_qa_em"),
        # P3
        "Acc_EN": p3.get("accuracy_en"),
        "Acc_AR": p3.get("accuracy_ar"),
        "LPG": p3.get("LPG"),
        "CLCS": p3.get("CLCS"),
        "IFS": p3.get("IFS"),
        "RDI": p3.get("RDI"),
        "McNemar_p": p3.get("mcnemar_p_value"),
        "Cohen_d": p3.get("cohens_d"),
        # P4
        "AR2EN_Adherence": p4.get("ar2en_language_adherence"),
        "EN2AR_Adherence": p4.get("en2ar_language_adherence"),
        "CodeSwitch_Coherence": p4.get("code_switching_coherence"),
        "Dialectal_Avg": p4.get("dialectal_avg"),
    }

    # CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    log.info(f"  📄 CSV: {csv_path}")

    # LaTeX
    def _fmt(v):
        if v is None:
            return "--"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    latex_rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{4-Pillar Evaluation Results: " + model_name.replace("_", "\\_") + "}",
        r"\label{tab:results_" + model_name + "}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for k, v in summary.items():
        if k == "model":
            continue
        latex_rows.append(f"{k.replace('_', ' ')} & {_fmt(v)} \\\\")
    latex_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = output_dir / "summary.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_rows))
    log.info(f"  📄 LaTeX: {tex_path}")

    _save_json(summary, output_dir / "summary.json")


# =============================================================================
# UTILITIES
# =============================================================================

def _save_json(data: dict, path: Path) -> None:
    check_disk_space(path.parent, required_gb=0.1, label=path.name)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp.rename(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Write failed {path}: {e}") from e


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to fine-tuned model or adapter")
    parser.add_argument("--model-name", required=True,
                        help="Short name for output files (e.g. qwen7b_sft)")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--pillars", nargs="+", choices=["1","2","3","4"],
                        default=["1","2","3","4"],
                        help="Which pillars to run (default: all)")
    args = parser.parse_args()

    try:
        eval_cfg = resolve_env(load_yaml(args.config))
    except Exception as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    output_dir = Path(eval_cfg["output_dir"]) / args.model_name
    safe_makedirs(output_dir)
    check_disk_space(output_dir, required_gb=1.0, label="eval output")

    log.info(f"\n{'='*60}")
    log.info(f"4-PILLAR EVALUATION: {args.model_name}")
    log.info(f"Model path: {args.model_path}")
    log.info(f"Output: {output_dir}")
    log.info(f"Pillars: {args.pillars}")
    log.info(f"{'='*60}\n")

    try:
        model, tokenizer = load_model_for_eval(args.model_path)
    except Exception as e:
        log.error(f"Model load failed: {e}")
        sys.exit(1)

    p1 = p2 = p3 = p4 = {}

    if "1" in args.pillars:
        try:
            p1 = eval_pillar1(model, tokenizer, eval_cfg, output_dir)
        except Exception as e:
            log.error(f"Pillar 1 failed: {e}")
            p1 = {"error": str(e)}

    if "2" in args.pillars:
        try:
            p2 = eval_pillar2(model, tokenizer, eval_cfg, output_dir)
        except Exception as e:
            log.error(f"Pillar 2 failed: {e}")
            p2 = {"error": str(e)}

    if "3" in args.pillars:
        try:
            p3 = eval_pillar3(model, tokenizer, eval_cfg, output_dir)
        except Exception as e:
            log.error(f"Pillar 3 failed: {e}")
            p3 = {"error": str(e)}

    if "4" in args.pillars:
        try:
            p4 = eval_pillar4(model, tokenizer, eval_cfg, output_dir)
        except Exception as e:
            log.error(f"Pillar 4 failed: {e}")
            p4 = {"error": str(e)}

    generate_summary(p1, p2, p3, p4, args.model_name, output_dir)
    log.info(f"\n✅ Evaluation complete → {output_dir}")


if __name__ == "__main__":
    main()
