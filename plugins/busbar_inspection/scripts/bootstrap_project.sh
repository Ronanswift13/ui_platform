#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target-project-dir> [project-name]"
  exit 1
fi

TARGET_DIR="$1"
PROJECT_NAME="${2:-$(basename "$TARGET_DIR")}" 
TEMPLATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$TARGET_DIR"
rsync -a   --exclude '.git'   --exclude '.DS_Store'   "$TEMPLATE_DIR"/ "$TARGET_DIR"/

for f in "$TARGET_DIR/CLAUDE.md" "$TARGET_DIR/PROJECT_CARD.md"; do
  if [[ -f "$f" ]]; then
    sed -i '' "s/{{PROJECT_NAME}}/${PROJECT_NAME}/g" "$f"
  fi
done

echo "Template copied to: $TARGET_DIR"
echo "Next steps:"
echo "1) Fill PROJECT_CARD.md"
echo "2) Fill .agent_skills/00_project_context.md"
echo "3) Customize validation commands in CLAUDE.md"
