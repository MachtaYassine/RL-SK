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
# Skull King RL Training with Auto-Resume (PPO Self-Play)
# ============================================================
# Handles the 8hr SLURM limit by:
# 1. Catching SIGUSR1 (sent 5min before timeout)
# 2. Requeuing the job automatically
# 3. Resuming from checkpoints/latest.pt on restart
# ============================================================

# --- Configuration ---
TOTAL_GAMES=100000
NUM_PLAYERS=4
WORKERS=4
DEVICE="cpu"
SAVE_DIR="checkpoints"
LOG_DIR="runs"

# --- Setup ---
mkdir -p slurm_logs

# --- Signal handler: requeue on timeout ---
requeue_job() {
    echo "$(date): Caught timeout signal, job will be requeued..."
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_job USR1

# --- Activate environment (adjust to your setup) ---
# source ~/miniconda3/bin/activate skullking
# module load python/3.10
# source ~/venv/bin/activate

echo "$(date): Starting training (Job ID: $SLURM_JOB_ID)"
echo "Games: $TOTAL_GAMES | Players: $NUM_PLAYERS | Workers: $WORKERS"
echo "Save: $SAVE_DIR | Logs: $LOG_DIR"

if [ -f "${SAVE_DIR}/latest.pt" ]; then
    echo "$(date): Found checkpoint, will auto-resume"
fi

# --- Run training (auto-detects latest.pt for resume) ---
python cli.py train \
    --games $TOTAL_GAMES \
    --players $NUM_PLAYERS \
    --workers $WORKERS \
    --device $DEVICE \
    --save-dir $SAVE_DIR \
    --log-dir $LOG_DIR &

# Wait for the python process (needed for signal trapping to work)
wait $!
echo "$(date): Training finished or interrupted"
