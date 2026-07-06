#!/bin/zsh

set -euo pipefail

ROOT="/Users/rowanwarneke/Desktop/chris gay"
LOG_FILE="$ROOT/delayed_git_commit_push.log"
COMMIT_MESSAGE="Update World Cup 2026 data and site outputs"

{
  echo "[$(date)] Starting delayed git commit/push job"
  cd "$ROOT"

  BRANCH="$(git branch --show-current)"
  echo "[$(date)] Branch: $BRANCH"
  echo "[$(date)] Sleeping for 10 minutes"
  sleep 600

  echo "[$(date)] Staging changes"
  git add -A

  if git diff --cached --quiet; then
    echo "[$(date)] No staged changes to commit"
  else
    echo "[$(date)] Creating commit"
    git commit -m "$COMMIT_MESSAGE"
  fi

  echo "[$(date)] Pushing to origin/$BRANCH"
  git push origin "$BRANCH"
  echo "[$(date)] Job complete"
} >> "$LOG_FILE" 2>&1
