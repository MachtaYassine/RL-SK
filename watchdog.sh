#!/bin/bash
#SBATCH --job-name=sk-watchdog
#SBATCH --output=slurm_logs/watchdog_%j.out
#SBATCH --error=slurm_logs/watchdog_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=512M

# ============================================================
# Watchdog: ensures training reaches 100k games
# Schedule this with: sbatch --begin=now+4hours --dependency=singleton watchdog.sh
# It checks if a training job is running; if not, resubmits one.
# Then resubmits itself to check again later.
# ============================================================

NTFY_TOPIC="skullking-rl-ymachta-8f3k"
notify() { curl -s -d "$1" "ntfy.sh/${NTFY_TOPIC}" > /dev/null 2>&1 || true; }

TRAIN_SCRIPT="run_experiment.sh"
TRAIN_JOB_NAME="skullking-rl"
CHECKPOINT="checkpoints/latest.pt"
TARGET_GAMES=100000

# Check if training is already done
if [ -f "$CHECKPOINT" ]; then
    GAMES_DONE=$(conda run --no-capture-output -n rlsk python -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=True)
print(ckpt.get('total_games', 0))
")
    echo "$(date): Games completed: $GAMES_DONE / $TARGET_GAMES"
    if [ "$GAMES_DONE" -ge "$TARGET_GAMES" ]; then
        notify "✅ Training COMPLETE! ${GAMES_DONE}/${TARGET_GAMES} games. Watchdog retiring."
        echo "$(date): Training complete! Watchdog retiring."
        exit 0
    fi
fi

# Check if a training job is already running or pending
RUNNING=$(squeue -u "$USER" -n "$TRAIN_JOB_NAME" -h | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "$(date): Training job already in queue, nothing to do."
else
    echo "$(date): No training job found, resubmitting..."
    notify "🔄 Watchdog: training job died at ${GAMES_DONE:-?}/${TARGET_GAMES} games, resubmitting..."
    sbatch "$TRAIN_SCRIPT"
fi

# Resubmit watchdog to check again in 4 hours
sbatch --begin=now+4hours watchdog.sh
echo "$(date): Watchdog will check again in 4 hours."
