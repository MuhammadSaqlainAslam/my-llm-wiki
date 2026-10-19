#!/bin/bash
# setup_cron.sh — Install daily cron job for the wiki ingestion agent
#
# Configure via environment variables (sensible defaults shown):
#   WIKI_VAULT  — path to the wiki repo   (default: this script's directory)
#   CONDA_SH    — path to conda's profile.d/conda.sh
#   CONDA_ENV   — conda environment name to activate
VAULT="${WIKI_VAULT:-$(cd "$(dirname "$0")" && pwd)}"
CONDA="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-base}"
LOG="$VAULT/cron.log"

# Build the cron command: source conda, activate env, then run agent
CRON_CMD="0 8 * * * source $CONDA && conda activate $CONDA_ENV && cd $VAULT && python3 agent.py daily --push >> $LOG 2>&1"

# Append only if not already present
if crontab -l 2>/dev/null | grep -q "agent.py daily"; then
    echo "Cron job already exists — no change."
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job added: agent runs daily at 8:00 AM"
fi

echo ""
echo "Current crontab:"
crontab -l
