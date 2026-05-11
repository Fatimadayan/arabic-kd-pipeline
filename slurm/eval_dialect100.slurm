#!/bin/bash
#SBATCH --job-name=kd_dialect100
#SBATCH --output=/data/datasets/user151/qwen-arabic-kd/logs/dialect100_%j.out
#SBATCH --error=/data/datasets/user151/qwen-arabic-kd/logs/dialect100_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate /data/datasets/user151/conda-envs/kd_env

python - << 'PYEOF'
import json
import random

# =========================
# 1. DIALECT ID (33 prompts)
# =========================
dialect_id = [
"شلونك اليوم؟",
"وايد حار اليوم",
"الحين بروح البحر",
"شلون الشغل؟",
"وايد تعبان",
"الحين نخلص ونمشي",
"شلون الدراسة معاك؟",
"وايد زحمة الشارع",
"الحين الجو تغير",
"شلون الأهل؟",
"وايد حلو المكان",
"الحين خل نبدأ",
"شلون الوضع؟",
"وايد صعب الموضوع",
"الحين عندي شغل",
"شلون الصحة؟",
"وايد سريع الإنترنت",
"الحين بنشوف الحل",
"شلون كان الامتحان؟",
"وايد غالي السعر",
"الحين الوقت مناسب",
"شلونك من زمان؟",
"وايد مبسوط اليوم",
"الحين خلصنا",
"شلون الأمور؟",
"وايد تعبنا اليوم",
"الحين نراجع",
"شلون المشروع؟",
"وايد ممتاز",
"الحين نبدأ الشغل",
"شلونك الحين؟",
"وايد واضح",
"الحين خل نحل المشكلة"
]

# =========================
# 2. NORMALIZATION (33 prompts)
# =========================
normalize = [
"حول للفصحى: شلونك",
"حول للفصحى: وايد حلو",
"حول للفصحى: الحين بروح",
"حول للفصحى: شلون الدراسة",
"حول للفصحى: وايد صعب",
"حول للفصحى: الحين نبدأ",
"حول للفصحى: شلون الوضع",
"حول للفصحى: وايد زحمة",
"حول للفصحى: الحين نخلص",
"حول للفصحى: شلونك اليوم",
"حول للفصحى: وايد غالي",
"حول للفصحى: الحين نراجع",
"حول للفصحى: شلون المشروع",
"حول للفصحى: وايد ممتاز",
"حول للفصحى: الحين الوقت مناسب",
"حول للفصحى: شلون الأمور",
"حول للفصحى: وايد تعبان",
"حول للفصحى: الحين خل نبدأ",
"حول للفصحى: شلون كان الامتحان",
"حول للفصحى: وايد واضح",
"حول للفصحى: الحين بنشوف",
"حول للفصحى: شلونك من زمان",
"حول للفصحى: وايد حار",
"حول للفصحى: الحين بروح السوق",
"حول للفصحى: شلون الصحة",
"حول للفصحى: وايد سريع",
"حول للفصحى: الحين خلصنا",
"حول للفصحى: شلون الشغل",
"حول للفصحى: وايد مبسوط",
"حول للفصحى: الحين نبدأ الشغل",
"حول للفصحى: شلونك الحين",
"حول للفصحى: وايد صعب الموضوع",
"حول للفصحى: الحين نحل المشكلة"
]

# =========================
# 3. CODE SWITCH (34 prompts)
# =========================
code_switch = [
"الحين نقدر deploy النظام؟",
"في مشكلة في system لازم نحلها",
"شلون الperformance مال التطبيق؟",
"وايد slow الresponse",
"الحين نعمل update للنظام",
"في bug في الكود لازم fix",
"شلون الAPI يشتغل؟",
"وايد good النتيجة",
"الحين نبدأ testing",
"في error في البرنامج",
"شلون الdatabase متصل؟",
"وايد fast النظام",
"الحين نdeploy النسخة",
"في مشكلة في server",
"شلون الfrontend؟",
"وايد clean الكود",
"الحين نعمل debug",
"في issue في login",
"شلون الbackend؟",
"وايد stable النظام",
"الحين نpush التحديث",
"في problem في network",
"شلون الUI؟",
"وايد smooth الأداء",
"الحين نعمل restart",
"في bug في API",
"شلون الdesign؟",
"وايد responsive الموقع",
"الحين نعمل test case",
"في crash في النظام",
"شلون الsecurity؟",
"وايد safe التطبيق",
"الحين نعمل deploy جديد",
"في latency عالي"
]

# =========================
# Combine dataset
# =========================
dataset = []

for q in dialect_id:
    dataset.append({"cat": "dialect_id", "q": q})

for q in normalize:
    dataset.append({"cat": "normalize", "q": q})

for q in code_switch:
    dataset.append({"cat": "code_switch", "q": q})

random.shuffle(dataset)

# =========================
# SAVE
# =========================
out_path = "/data/datasets/user151/qwen-arabic-kd/dialect100_prompts.json"

with open(out_path, "w") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Saved 100 prompts to:", out_path)
print("Total samples:", len(dataset))

PYEOF
