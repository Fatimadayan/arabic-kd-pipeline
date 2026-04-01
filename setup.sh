#!/bin/bash
# =============================================================================
# setup.sh
# ONE-SHOT ENVIRONMENT SETUP FOR BENEFIT LAB (UOB)
#
# Run this ONCE after SSH login:
#   bash setup.sh
#
# What it does:
#   1. Creates all project directories under /data/datasets/$USER/
#   2. Sets HuggingFace cache env vars (adds to ~/.bashrc)
#   3. Creates Python virtual environment
#   4. Installs all dependencies
#   5. Runs safety check to confirm everything is OK
#
# CLUSTER SAFETY:
#   - Never writes to $HOME (5GB quota)
#   - All data goes to /data/datasets/$USER/ (large quota)
#   - Disk checked before pip install
#   - set -euo pipefail: stops on any error
# =============================================================================

set -euo pipefail

# ── Colours for readability ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
fail() { echo -e "${RED}❌ $*${NC}"; exit 1; }

echo "=============================================="
echo "  Benefit Lab Setup — Arabic KD Pipeline"
echo "  User: $USER"
echo "  Date: $(date)"
echo "=============================================="
echo ""

# =============================================================================
# 1. VALIDATE WE ARE ON THE RIGHT MACHINE
# =============================================================================
echo "1️⃣  Validating cluster environment..."

if [ ! -d "/data/datasets" ]; then
    fail "/data/datasets does not exist. Are you SSH'd into Benefit Lab?"
fi

DATA_ROOT="/data/datasets/$USER"
ok "Data root: ${DATA_ROOT}"

# =============================================================================
# 2. CHECK DISK SPACE (need ~50GB free for full experiment)
# =============================================================================
echo ""
echo "2️⃣  Checking disk space..."

AVAILABLE_GB=$(df -BG /data/datasets | awk 'NR==2 {gsub("G",""); print $4}')
REQUIRED_GB=50

echo "   Available: ${AVAILABLE_GB}G"
echo "   Required:  ${REQUIRED_GB}G"

if [ "${AVAILABLE_GB}" -lt "${REQUIRED_GB}" ]; then
    warn "Only ${AVAILABLE_GB}G available. Recommend at least ${REQUIRED_GB}G."
    warn "Continuing — but monitor disk with: df -h /data/datasets"
else
    ok "Disk space OK (${AVAILABLE_GB}G available)"
fi

# Check home quota (warn if >80% full)
HOME_USED=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $3}')
HOME_TOTAL=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $2}')
echo "   Home quota: ${HOME_USED}G / ${HOME_TOTAL}G used"
if [ "${HOME_USED}" -ge 4 ]; then
    warn "Home directory near 5GB quota! Keep ALL work in /data/datasets/$USER/"
fi

# =============================================================================
# 3. CREATE DIRECTORY STRUCTURE
# =============================================================================
echo ""
echo "3️⃣  Creating project directories..."

PROJECT="${DATA_ROOT}/qwen-arabic-kd"

mkdir -p "${PROJECT}"/{configs,scripts,slurm,data,checkpoints,results,analysis,logs}
mkdir -p "${DATA_ROOT}/.cache/huggingface/transformers"
mkdir -p "${DATA_ROOT}/.cache/huggingface/datasets"
mkdir -p "${DATA_ROOT}/.cache/torch"

ok "Directories created under ${PROJECT}"

# =============================================================================
# 4. SET ENVIRONMENT VARIABLES
# =============================================================================
echo ""
echo "4️⃣  Configuring environment variables..."

# Check if already set
if grep -q "HF_HOME=/data/datasets" ~/.bashrc 2>/dev/null; then
    warn "Env vars already in ~/.bashrc — skipping (already configured)"
else
    cat >> ~/.bashrc << 'EOF'

# ── Arabic KD Pipeline — Benefit Lab ──────────────────────────────────────
export DATA_ROOT="/data/datasets/$USER"
export PROJECT_ROOT="/data/datasets/$USER/qwen-arabic-kd"
export HF_HOME="/data/datasets/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/datasets/$USER/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/data/datasets/$USER/.cache/huggingface/datasets"
export TORCH_HOME="/data/datasets/$USER/.cache/torch"
export PYTHONPATH="/data/datasets/$USER/qwen-arabic-kd:$PYTHONPATH"
# ── End Arabic KD Pipeline ────────────────────────────────────────────────
EOF
    ok "Env vars added to ~/.bashrc"
fi

# Export for current session
export DATA_ROOT="/data/datasets/$USER"
export PROJECT_ROOT="${PROJECT}"
export HF_HOME="/data/datasets/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="/data/datasets/$USER/.cache/torch"

# =============================================================================
# 5. CHECK PYTHON
# =============================================================================
echo ""
echo "5️⃣  Checking Python..."

PYTHON_BIN=""
for candidate in python3.10 python3.11 python3.9 python3; do
    if command -v "${candidate}" &>/dev/null; then
        PYTHON_BIN="${candidate}"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    fail "No Python 3 found. Try: module load python/3.10"
fi

PYTHON_VERSION=$(${PYTHON_BIN} --version)
ok "Python: ${PYTHON_VERSION} (${PYTHON_BIN})"

# =============================================================================
# 6. CREATE VIRTUAL ENVIRONMENT (inside /data/datasets/$USER — not $HOME)
# =============================================================================
echo ""
echo "6️⃣  Setting up virtual environment..."

VENV_PATH="${DATA_ROOT}/venvs/kd_env"

if [ -d "${VENV_PATH}" ]; then
    warn "Venv already exists at ${VENV_PATH} — skipping creation"
else
    mkdir -p "${DATA_ROOT}/venvs"
    ${PYTHON_BIN} -m venv "${VENV_PATH}"
    ok "Venv created: ${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
ok "Venv activated"

# Add activation to bashrc if not present
if ! grep -q "kd_env/bin/activate" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Auto-activate KD venv (comment out if unwanted)" >> ~/.bashrc
    echo "source ${VENV_PATH}/bin/activate" >> ~/.bashrc
    ok "Auto-activation added to ~/.bashrc"
fi

# =============================================================================
# 7. INSTALL DEPENDENCIES
# =============================================================================
echo ""
echo "7️⃣  Installing Python dependencies..."
echo "   Estimated time: 5-10 minutes"
echo "   Estimated disk: ~3-5GB in ${VENV_PATH}"

REQUIREMENTS="${PROJECT}/requirements.txt"
if [ ! -f "${REQUIREMENTS}" ]; then
    fail "requirements.txt not found at ${REQUIREMENTS}. Upload project files first."
fi

pip install --upgrade pip --quiet
pip install -r "${REQUIREMENTS}" --quiet \
    || fail "pip install failed. Check network and disk space."

ok "All dependencies installed"

# =============================================================================
# 8. VERIFY GPU ACCESS
# =============================================================================
echo ""
echo "8️⃣  Checking GPU..."

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    ok "GPU detected"
else
    warn "nvidia-smi not found — you may be on a login node. GPUs available via SLURM."
fi

# =============================================================================
# 9. RUN SAFETY CHECK
# =============================================================================
echo ""
echo "9️⃣  Running safety check..."

if [ -f "${PROJECT}/scripts/00_safety_check.sh" ]; then
    bash "${PROJECT}/scripts/00_safety_check.sh"
else
    warn "00_safety_check.sh not found — run it manually after uploading all files."
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "  SETUP COMPLETE"
echo "=============================================="
echo ""
echo "  Project:  ${PROJECT}"
echo "  Venv:     ${VENV_PATH}"
echo "  HF Cache: ${HF_HOME}"
echo ""
echo "  NEXT STEPS:"
echo "  1. source ~/.bashrc"
echo "  2. cd ${PROJECT}"
echo "  3. bash scripts/00_safety_check.sh"
echo "  4. python scripts/run_experiment.py --dry-run"
echo "  5. sbatch slurm/teacher_gen.slurm"
echo ""
echo "  MONITOR DISK:   df -h /data/datasets"
echo "  MONITOR JOBS:   squeue -u $USER"
echo "  MONITOR GPU:    nvidia-smi (on GPU node)"
echo "=============================================="
