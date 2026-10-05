#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/streamlit ]]; then
  echo "Environment not configured. Run: bash scripts/setup_mac.sh"
  exit 1
fi
source .venv/bin/activate
exec streamlit run app.py
