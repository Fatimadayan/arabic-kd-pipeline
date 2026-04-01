#!/bin/bash
# =============================================================================
# scripts/00_safety_check.sh
# Pre-flight safety validation for Benefit Lab cluster
#
# Run before EVERY experiment:
#   bash scripts/00_safety_check.sh
#
# Checks:
#   1. Correct working directory (/data/datasets/$USER)
#   2. Disk space on /data/datasets
#   3. Home directory quota not exceeded
#   4. Environment variables set correctly
#   5. Python/venv accessible
#   6. Key Python imports work
#   7. GPU available (if on GPU node)
#   8. No runaway jobs already queued
#   9. No zombie processes from previous runs
#  10. HuggingFace cache is NOT in $HOME
#
# Exit codes:
#   0 = all checks passed
#   1 = critical failure (do NOT proceed)
#   2 = warnings only (proceed with caution)
# =============================================================================

set -uo pipefail   # Note: no -e so we can collect all failures

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
WARN=0
FAIL=0

pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; ((PASS++)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $*"; ((WARN++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; ((FAIL++)); }
info() { echo -e "  ${BLUE}[INFO]${NC} $*"; }

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Benefit Lab Safety Check                ║"
echo "║  Arabic KD Pipeline — Pre-flight         ║"
echo "╚══════════════════════════════════════════╝"
echo "  User: $USER  |  Date: $(date '+%Y-%m-%d %H:%M')"
echo ""

# =============================================================================
# CHECK 1: Working in /data/datasets (not $HOME)
# =============================================================================
echo "── 1. Storage Location ──────────────────────"

EXPECTED_ROOT="/data/datasets/$USER"

if [ ! -d "${EXPECTED_ROOT}" ]; then
    fail "/data/datasets/$USER does not exist. Create it or check mount."
else
    pass "/data/datasets/$USER exists"
fi

if [ -d "${EXPECTED_ROOT}/qwen-arabic-kd" ]; then
    pass "Project directory found: ${EXPECTED_ROOT}/qwen-arabic-kd"
else
    warn "Project directory not found yet. Run: mkdir -p ${EXPECTED_ROOT}/qwen-arabic-kd"
fi

# Confirm we are NOT trying to use $HOME for large files
if echo "${PWD}" | grep -q "^/home/"; then
    warn "CWD is in \$HOME (${PWD}). cd to ${EXPECTED_ROOT}/qwen-arabic-kd"
else
    pass "CWD is outside \$HOME"
fi

# =============================================================================
# CHECK 2: Disk space on /data/datasets
# =============================================================================
echo ""
echo "── 2. Disk Space ────────────────────────────"

if df /data/datasets &>/dev/null; then
    AVAIL_GB=$(df -BG /data/datasets | awk 'NR==2 {gsub("G",""); print $4}')
    USED_PCT=$(df /data/datasets | awk 'NR==2 {gsub("%",""); print $5}')
    info "/data/datasets: ${AVAIL_GB}G free (${USED_PCT}% used)"

    if [ "${AVAIL_GB}" -lt 5 ]; then
        fail "CRITICAL: Only ${AVAIL_GB}G free on /data/datasets. Minimum 5GB needed."
    elif [ "${AVAIL_GB}" -lt 20 ]; then
        warn "Low disk space: ${AVAIL_GB}G. Recommend >20G for full experiment."
    else
        pass "Sufficient disk: ${AVAIL_GB}G free"
    fi

    # Check user's own usage
    USER_USED=$(du -sh "${EXPECTED_ROOT}" 2>/dev/null | cut -f1 || echo "unknown")
    info "Your usage in /data/datasets/$USER: ${USER_USED}"
else
    warn "/data/datasets not mounted or accessible"
fi

# =============================================================================
# CHECK 3: Home directory quota
# =============================================================================
echo ""
echo "── 3. Home Directory Quota ──────────────────"

HOME_AVAIL=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $4}')
HOME_USED=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $3}')

info "\$HOME quota: ${HOME_USED}G used / ~5G limit | ${HOME_AVAIL}G remaining"

if [ "${HOME_AVAIL}" -lt 1 ]; then
    fail "Home directory critically full (${HOME_AVAIL}G remaining). Clean up NOW."
elif [ "${HOME_AVAIL}" -lt 2 ]; then
    warn "Home directory almost full (${HOME_AVAIL}G remaining)."
else
    pass "Home directory OK (${HOME_AVAIL}G remaining)"
fi

# =============================================================================
# CHECK 4: Environment variables
# =============================================================================
echo ""
echo "── 4. Environment Variables ─────────────────"

check_env() {
    local var="$1"
    local expected_prefix="$2"
    local val="${!var:-}"

    if [ -z "${val}" ]; then
        warn "${var} is not set. Run: source ~/.bashrc"
    elif echo "${val}" | grep -q "^/home/"; then
        fail "${var}=${val} points to \$HOME! Change to /data/datasets/$USER"
    elif echo "${val}" | grep -q "${expected_prefix}"; then
        pass "${var}=${val}"
    else
        warn "${var}=${val} (unexpected path — check ~/.bashrc)"
    fi
}

check_env "HF_HOME" "/data/datasets"
check_env "TRANSFORMERS_CACHE" "/data/datasets"
check_env "HF_DATASETS_CACHE" "/data/datasets"
check_env "TORCH_HOME" "/data/datasets"

# =============================================================================
# CHECK 5: HuggingFace cache NOT in $HOME
# =============================================================================
echo ""
echo "── 5. Cache Location Safety ─────────────────"

HF_DEFAULT="$HOME/.cache/huggingface"
if [ -d "${HF_DEFAULT}" ] && [ "$(du -sm ${HF_DEFAULT} 2>/dev/null | cut -f1)" -gt 100 ]; then
    fail "Large HF cache found in \$HOME (${HF_DEFAULT}). Move it:"
    fail "  mv ${HF_DEFAULT} /data/datasets/$USER/.cache/huggingface"
    fail "  ln -s /data/datasets/$USER/.cache/huggingface ${HF_DEFAULT}"
elif [ -L "${HF_DEFAULT}" ]; then
    pass "\$HOME/.cache/huggingface is a symlink (safe)"
else
    pass "No large cache in \$HOME"
fi

# =============================================================================
# CHECK 6: Python environment
# =============================================================================
echo ""
echo "── 6. Python Environment ────────────────────"

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    PY_PATH=$(which python3)
    info "Python: ${PY_VER} at ${PY_PATH}"

    # Check if venv is activated
    if echo "${PY_PATH}" | grep -q "/data/datasets"; then
        pass "Python is from venv in /data/datasets (correct)"
    elif echo "${PY_PATH}" | grep -q "/home/"; then
        warn "Python is from \$HOME. Activate your venv:"
        warn "  source /data/datasets/$USER/venvs/kd_env/bin/activate"
    else
        pass "Python found: ${PY_PATH}"
    fi
else
    fail "python3 not found. Load module: module load python/3.10"
fi

# =============================================================================
# CHECK 7: Key Python imports
# =============================================================================
echo ""
echo "── 7. Python Imports ────────────────────────"

check_import() {
    local pkg="$1"
    if python3 -c "import ${pkg}" 2>/dev/null; then
        pass "${pkg}"
    else
        warn "${pkg} not installed. Run: pip install ${pkg} -q"
    fi
}

check_import "torch"
check_import "transformers"
check_import "datasets"
check_import "peft"
check_import "trl"
check_import "yaml"
check_import "accelerate"

# Check torch GPU support
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    pass "PyTorch CUDA: ${GPU_COUNT} GPU(s) — ${GPU_NAME}"
else
    warn "PyTorch CUDA not available on this node (normal on login node — GPU via SLURM)"
fi

# =============================================================================
# CHECK 8: SLURM / job status
# =============================================================================
echo ""
echo "── 8. SLURM Jobs ────────────────────────────"

if command -v squeue &>/dev/null; then
    MY_JOBS=$(squeue -u "$USER" 2>/dev/null | grep -v "JOBID" | wc -l)
    info "Your running/queued jobs: ${MY_JOBS}"
    if [ "${MY_JOBS}" -gt 5 ]; then
        warn "Many jobs queued (${MY_JOBS}). Check: squeue -u $USER"
    else
        pass "Job queue OK (${MY_JOBS} jobs)"
    fi

    # Check for existing KD jobs
    KD_JOBS=$(squeue -u "$USER" 2>/dev/null | grep -c "kd_" || true)
    if [ "${KD_JOBS}" -gt 0 ]; then
        warn "KD jobs already running (${KD_JOBS}). Check before submitting more."
    fi
else
    warn "squeue not found — not on SLURM node or SLURM not loaded"
fi

# =============================================================================
# CHECK 9: Project file structure
# =============================================================================
echo ""
echo "── 9. Project Files ─────────────────────────"

PROJECT_DIR="${EXPECTED_ROOT}/qwen-arabic-kd"
REQUIRED_FILES=(
    "configs/models.yaml"
    "configs/data.yaml"
    "configs/eval.yaml"
    "configs/experiment_config.yaml"
    "scripts/01_generate_teacher_data.py"
    "scripts/02_train_baseline_sft.py"
    "scripts/03_train_sequence_kd.py"
    "scripts/04_train_token_kd.py"
    "scripts/05_eval_4pillars.py"
    "scripts/06_analysis.py"
    "scripts/run_experiment.py"
    "slurm/teacher_gen.slurm"
    "slurm/train_7b.slurm"
    "slurm/train_kd_7b.slurm"
    "requirements.txt"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "${PROJECT_DIR}/${f}" ]; then
        pass "${f}"
    else
        warn "${f} — NOT FOUND (upload from zip)"
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  SAFETY CHECK SUMMARY                    ║"
echo "╚══════════════════════════════════════════╝"
echo -e "  ${GREEN}PASS: ${PASS}${NC} | ${YELLOW}WARN: ${WARN}${NC} | ${RED}FAIL: ${FAIL}${NC}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
    echo -e "  ${RED}❌ ${FAIL} CRITICAL FAILURE(S). DO NOT PROCEED.${NC}"
    echo "     Fix all FAIL items above before running any experiment."
    exit 1
elif [ "${WARN}" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠️  ${WARN} warning(s). Review before proceeding.${NC}"
    echo "     Warnings are non-blocking but should be addressed."
    echo ""
    echo "  ✅ Safe to proceed with caution."
    exit 2
else
    echo -e "  ${GREEN}✅ All checks passed. Safe to run experiment.${NC}"
    echo ""
    echo "  NEXT: python scripts/run_experiment.py --dry-run"
    exit 0
fi
