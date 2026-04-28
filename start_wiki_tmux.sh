#!/bin/bash

SESSION="wiki"
WIKI_DIR="/work/HHRI-AI/Saqlain/my-wiki"

# Don't create a duplicate session
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach-session -t "$SESSION"
    exit 0
fi

# --- Window 1: Editor (Home.md) ---
tmux new-session -d -s "$SESSION" -n "edit" -c "$WIKI_DIR/wiki"
tmux send-keys -t "$SESSION:edit" "vim Home.md" Enter

# --- Window 2: Browse (wiki file list) ---
tmux new-window -t "$SESSION" -n "browse" -c "$WIKI_DIR/wiki"
# Left pane: live file list
tmux send-keys -t "$SESSION:browse" "watch -n2 'ls -1 *.md'" Enter
# Right pane: raw PDFs
tmux split-window -t "$SESSION:browse" -h -c "$WIKI_DIR/raw"
tmux send-keys -t "$SESSION:browse.right" "ls -1 *.pdf" Enter

# --- Window 3: Claude Code ---
tmux new-window -t "$SESSION" -n "claude" -c "$WIKI_DIR"
tmux send-keys -t "$SESSION:claude" "source activate_claude_w_Azure_API.sh" Enter

# Start on the editor window
tmux select-window -t "$SESSION:edit"

tmux attach-session -t "$SESSION"
