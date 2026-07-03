#!/usr/bin/env bash
#
# Launch the Medical Report Information Extractor from the project's .venv312 —
# the environment that has BOTH the app stack AND the on-device Lift engine
# (datalab-to/lift + PyTorch).
#
# Why this exists: running a bare `streamlit run app.py` can resolve `streamlit`
# to a different interpreter on your PATH (e.g. Homebrew python@3.12) that does
# not have `lift` installed. When that happens the Lift engine only shows its
# "install the lift package" hint. This script always uses .venv312/bin/python,
# so the Lift engine is available.
#
# Usage:
#   ./run.sh                          # launch on the default port (8501)
#   ./run.sh --server.port 8502       # any extra args pass through to streamlit
#
set -euo pipefail

# Resolve the project dir from this script's real location, so it works whether
# it's run from the Desktop path or the firmlinked container copy.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="$DIR/.venv312"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found." >&2
  echo "Create the env and install dependencies:" >&2
  echo "  python3 -m venv \"$VENV\"" >&2
  echo "  \"$VENV/bin/python\" -m pip install -r \"$DIR/requirements.txt\" 'lift-pdf[hf]'" >&2
  exit 1
fi

# Preflight: streamlit is required; lift is optional (its absence only disables
# the Lift engine, the rest of the app still runs).
if ! "$PY" -c 'import streamlit' 2>/dev/null; then
  echo "error: streamlit is not installed in .venv312." >&2
  echo "  \"$PY\" -m pip install -r \"$DIR/requirements.txt\"" >&2
  exit 1
fi
if "$PY" -c 'import lift' 2>/dev/null; then
  echo "Lift engine:  available (on-device datalab-to/lift)"
else
  echo "Lift engine:  NOT installed  ->  \"$PY\" -m pip install 'lift-pdf[hf]'" >&2
fi
echo "Interpreter:  $("$PY" -c 'import sys; print(sys.executable, "(Python", "%d.%d.%d)" % sys.version_info[:3])')"
echo "Tip: stop any other Streamlit server first, or pass --server.port to use another port."
echo

# Run from the project dir; exec so Ctrl-C / signals go straight to Streamlit.
cd "$DIR"
exec "$PY" -m streamlit run "$DIR/app.py" "$@"
