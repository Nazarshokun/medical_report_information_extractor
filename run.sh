#!/usr/bin/env bash
#
# Launch the Medical Report Information Extractor from the project's single venv
# (.venv-surya, Python 3.12) — it holds the whole app stack AND on-device Surya
# OCR (datalab-to/surya via llama.cpp/Metal, in-process). Python 3.12 is required
# because Surya's pinned wheels have no Python 3.14 build.
#
# Why this exists: a bare `streamlit run app.py` can resolve `streamlit` to a
# different interpreter on your PATH that lacks the deps. This script always uses
# the project venv, so everything (including Surya) is available.
#
# Usage:
#   ./run.sh                       # launch on the default port (8501)
#   ./run.sh --server.port 8502    # extra args pass through to streamlit
#
set -euo pipefail

# Resolve the project dir from this script's real location, so it works whether
# it's run from the Desktop path or the firmlinked container copy.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="$DIR/.venv-surya"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found." >&2
  echo "Create the env (Python 3.10-3.13) and install dependencies:" >&2
  echo "  brew install llama.cpp" >&2
  echo "  python3.12 -m venv \"$VENV\"" >&2
  echo "  \"$PY\" -m pip install -r \"$DIR/requirements.txt\" surya-ocr" >&2
  exit 1
fi

# Preflight: streamlit is required; Surya is optional (its absence only hides the
# "Surya OCR" PDF modes, the rest of the app still runs).
if ! "$PY" -c 'import streamlit' 2>/dev/null; then
  echo "error: streamlit is not installed in $VENV." >&2
  echo "  \"$PY\" -m pip install -r \"$DIR/requirements.txt\"" >&2
  exit 1
fi
if "$PY" -c 'import surya' 2>/dev/null \
   && { command -v llama-server >/dev/null 2>&1 || [[ -x /opt/homebrew/bin/llama-server ]]; }; then
  echo "Surya OCR:    available (on-device, llama.cpp/Metal)"
else
  echo "Surya OCR:    NOT ready  ->  brew install llama.cpp; \"$PY\" -m pip install surya-ocr" >&2
fi
echo "Interpreter:  $("$PY" -c 'import sys; print(sys.executable, "(Python", "%d.%d.%d)" % sys.version_info[:3])')"
echo "Tip: stop any other Streamlit server first, or pass --server.port to use another port."
echo

# Run from the project dir; exec so Ctrl-C / signals go straight to Streamlit.
cd "$DIR"
exec "$PY" -m streamlit run "$DIR/app.py" "$@"
