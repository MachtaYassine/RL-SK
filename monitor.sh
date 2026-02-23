#!/bin/bash
# Quick monitoring helper for headless SLURM training
# Usage: ./monitor.sh

SAVE_DIR="checkpoints"

echo "=== Job Status ==="
squeue -u "$USER" -n skullking-rl

echo ""
echo "=== Latest Checkpoint ==="
if [ -f "${SAVE_DIR}/latest.pt" ]; then
    python -c "
import torch
ckpt = torch.load('${SAVE_DIR}/latest.pt', map_location='cpu', weights_only=True)
total_target = 100000
games = ckpt.get('total_games', 0)
elo = ckpt.get('player_elo', 1200)
print(f'Games: {games}/{total_target} ({100*games/total_target:.1f}%)')
print(f'ELO: {elo:.0f}')
"
else
    echo "No checkpoint yet"
fi

echo ""
echo "=== Last 10 Log Lines ==="
LATEST_LOG=$(ls -t slurm_logs/skullking_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -10 "$LATEST_LOG"
else
    echo "No SLURM logs yet"
fi

echo ""
echo "=== TensorBoard ==="
echo "To view remotely:  ssh -L 6006:localhost:6006 <cluster>"
echo "Then run:          tensorboard --logdir runs"
