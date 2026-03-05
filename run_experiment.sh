#!/bin/bash
#SBATCH --job-name=skullking-rl
#SBATCH --output=slurm_logs/skullking_%j.out
#SBATCH --error=slurm_logs/skullking_%j.err
#SBATCH --time=07:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --signal=B:USR1@300

# ============================================================
# Skull King RL Training — Best HPO Config (Trial #8)
# ============================================================
# Auto-resume strategy:
# 1. SIGUSR1 (5min before timeout) → requeue this job
# 2. On restart, cli.py auto-loads checkpoints/latest.pt
# 3. Training continues until 100k games reached
# ============================================================

# --- Best HPO Config (Trial #8, ELO=1120.7) ---
TOTAL_GAMES=100000
BATCH_SIZE=64
CLIP_EPS=0.115
ENTROPY_COEF=0.042
GAE_LAMBDA=0.964
GAMMA=0.994
HIDDEN_DIM=128
POLICY_EPOCHS=5
POLICY_LR=0.000224
VALUE_EPOCHS=4
VALUE_LR=0.000730
WORKERS=1
DEVICE="cpu"
SAVE_DIR="checkpoints"
LOG_DIR="runs"

# --- Notifications ---
NTFY_TOPIC="skullking-rl-ymachta-8f3k"  # subscribe to this in ntfy app
notify() { curl -s -d "$1" "ntfy.sh/${NTFY_TOPIC}" > /dev/null 2>&1 || true; }

# --- Setup ---
mkdir -p slurm_logs

# --- Signal handler: requeue on timeout ---
requeue_job() {
    echo "$(date): Caught timeout signal, requeuing job $SLURM_JOB_ID ..."
    notify "⏰ SLURM timeout, requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_job USR1

# Report current progress on start
GAMES_SO_FAR=0
if [ -f "${SAVE_DIR}/latest.pt" ]; then
    GAMES_SO_FAR=$(conda run --no-capture-output -n rlsk python -c "
import torch
ckpt = torch.load('${SAVE_DIR}/latest.pt', map_location='cpu', weights_only=True)
print(ckpt.get('total_games', 0))
" 2>/dev/null || echo 0)
    echo "$(date): Resuming from game $GAMES_SO_FAR"
fi

notify "🚀 Training started (job $SLURM_JOB_ID) — ${GAMES_SO_FAR}/${TOTAL_GAMES} games done"

echo "$(date): Starting training (Job ID: $SLURM_JOB_ID)"
echo "Config: batch=$BATCH_SIZE hidden=$HIDDEN_DIM gamma=$GAMMA entropy=$ENTROPY_COEF"

# --- Run training ---
conda run --no-capture-output -n rlsk python cli.py train \
    --games $TOTAL_GAMES \
    --vary-players \
    --policy-lr $POLICY_LR \
    --value-lr $VALUE_LR \
    --clip-eps $CLIP_EPS \
    --entropy-coef $ENTROPY_COEF \
    --gamma $GAMMA \
    --gae-lambda $GAE_LAMBDA \
    --hidden-dim $HIDDEN_DIM \
    --policy-epochs $POLICY_EPOCHS \
    --value-epochs $VALUE_EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --device $DEVICE \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR &

# Wait for python process (needed for signal trapping)
wait $!
EXIT_CODE=$?

# --- Report final status ---
FINAL_GAMES=0
if [ -f "${SAVE_DIR}/latest.pt" ]; then
    FINAL_GAMES=$(conda run --no-capture-output -n rlsk python -c "
import torch
ckpt = torch.load('${SAVE_DIR}/latest.pt', map_location='cpu', weights_only=True)
print(ckpt.get('total_games', 0))
" 2>/dev/null || echo 0)
fi

if [ "$FINAL_GAMES" -ge "$TOTAL_GAMES" ]; then
    notify "✅ Training COMPLETE! ${FINAL_GAMES}/${TOTAL_GAMES} games done"
elif [ $EXIT_CODE -ne 0 ] && [ -f "${SAVE_DIR}/latest.pt" ]; then
    notify "⚠️ Training interrupted at ${FINAL_GAMES}/${TOTAL_GAMES} games (exit $EXIT_CODE), resubmitting..."
    sbatch "$0"
fi

echo "$(date): Training finished or interrupted (exit code: $EXIT_CODE, games: $FINAL_GAMES)"
