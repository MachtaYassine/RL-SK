#!/bin/bash
#SBATCH --job-name=skullking-hpo
#SBATCH --output=slurm_logs/hpo_%j.out
#SBATCH --error=slurm_logs/hpo_%j.err
#SBATCH --time=07:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --signal=B:USR1@300

# ============================================================
# Skull King HPO with Optuna
# ============================================================
# Each trial runs a short training (10k games) with different
# hyperparameters. Optuna learns which HP regions are promising.
#
# Supports auto-resume: resubmit the same script and it picks
# up from the SQLite DB where it left off.
#
# To run multiple HPO workers in parallel:
#   for i in $(seq 1 4); do sbatch run_hpo.sh; done
# They share the same DB and coordinate via Optuna.
# ============================================================

TRIALS=50
GAMES_PER_TRIAL=10000
WORKERS=1
DEVICE="cpu"
OUTPUT_DIR="hpo_results"

mkdir -p slurm_logs "$OUTPUT_DIR"

# Requeue on timeout (picks up remaining trials from DB)
requeue_job() {
    echo "$(date): Timeout signal, requeuing..."
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_job USR1

echo "$(date): Starting HPO (Job ID: $SLURM_JOB_ID)"
echo "Trials: $TRIALS | Games/trial: $GAMES_PER_TRIAL"

conda run --no-capture-output -n rlsk python hpo.py \
    --trials $TRIALS \
    --games-per-trial $GAMES_PER_TRIAL \
    --workers $WORKERS \
    --device $DEVICE \
    --output-dir $OUTPUT_DIR &

wait $!
echo "$(date): HPO finished or interrupted"
