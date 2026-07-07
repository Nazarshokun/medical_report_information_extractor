# Deployment Guide — macOS & Windows

How to run the **Medical Report Information Extractor** on a laptop. Everything
runs **locally**: on-device OCR + a local (or cloud) LLM, so medical reports can
stay on the machine. See the [PHI / privacy note](#phi--privacy-note) at the end.

> **TL;DR (already set up):**
> - **macOS:** `./run.sh`
> - **Windows:** `.\run.ps1`
>
> First-time setup is below. The app opens at **http://localhost:8501**.

---

## What you're deploying

| Piece | What it is | Required? |
|---|---|---|
| **Streamlit app** (`app.py`) | The UI + extraction pipeline | Yes |
| **LLM** (extraction) | Local via **LM Studio** / **Ollama**, or cloud **OpenAI** / **Anthropic** | Yes (pick one) |
| **Surya OCR** (`surya-ocr` + `llama.cpp`) | On-device OCR for scanned PDFs, ~3 GB model | Recommended, optional |
| **Native PDF text** (`PyMuPDF`) | Text from born-digital PDFs (no OCR) | Built in |
| **Apple Vision OCR** (`ocrmac`) | Extra OCR mode, **macOS only** | Optional |
| **`ocrmypdf` + `tesseract`** | Alternative scanned-PDF OCR fallback | Optional |

The whole stack lives in **one virtual environment: `.venv-surya`**, on
**Python 3.10–3.13** (3.12 recommended). **Not Python 3.14** — Surya's pinned
`opencv`/wheels have no 3.14 build.

> The venv and `ocr_cache/` are **git-ignored** — they are not copied with the
> project. You recreate the venv on each laptop (steps below).

---

## Prerequisites (both platforms)

1. **Python 3.12** (3.10–3.13 all work; **not 3.14**).
2. **One LLM source:**
   - **LM Studio** (recommended, free, macOS + Windows) — runs a local model, or
   - **Ollama** (local), or
   - an **OpenAI** / **Anthropic** API key (cloud — see the PHI note).
3. **For Surya OCR (optional):** the `llama.cpp` **`llama-server`** binary.
4. **The project folder** on the laptop. Copy the whole
   `medical_report_information_extractor/` folder over, or `git clone` it if you
   keep it in a repo. It must contain `app.py`, `requirements.txt`, `run.sh`,
   `run.ps1`, `config/`, and `pages/`.

---

## Part A — macOS (Apple Silicon)

### 1. Install system tools with Homebrew

```bash
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.12 + llama.cpp (for Surya OCR)
brew install python@3.12 llama.cpp

# OPTIONAL — scanned-PDF fallback OCR (only if you want the ocrmypdf path)
brew install ocrmypdf tesseract
```

### 2. Create the venv and install dependencies

```bash
cd /path/to/medical_report_information_extractor

python3.12 -m venv .venv-surya
./.venv-surya/bin/python -m pip install --upgrade pip
./.venv-surya/bin/python -m pip install -r requirements.txt surya-ocr
```

### 3. Run

```bash
chmod +x run.sh        # first time only
./run.sh               # opens http://localhost:8501
# ./run.sh --server.port 8502   # use another port
```

`run.sh` prints whether **Surya OCR** is ready and which interpreter it used.
The **first** OCR downloads the ~3 GB Surya model and starts `llama-server`
automatically (one-time; kept warm afterward).

---

## Part B — Windows 10 / 11

### 1. Install Python 3.12

Download from <https://www.python.org/downloads/> and **check
"Add python.exe to PATH"** during install. Verify in PowerShell:

```powershell
py -3.12 --version      # should print Python 3.12.x
```

### 2. Create the venv and install dependencies

```powershell
cd C:\path\to\medical_report_information_extractor

py -3.12 -m venv .venv-surya
.\.venv-surya\Scripts\python -m pip install --upgrade pip
.\.venv-surya\Scripts\python -m pip install -r requirements.txt surya-ocr
```

> `ocrmac` (Apple Vision) is skipped automatically on Windows — that's expected.

### 3. (Optional) Enable Surya OCR on Windows

Surya needs a **`llama-server.exe`**. Windows has no Homebrew, so:

1. Download a **llama.cpp** Windows release from
   <https://github.com/ggml-org/llama.cpp/releases> (pick a `bin-win-*.zip` —
   CUDA build if you have an NVIDIA GPU, otherwise the CPU/Vulkan build).
2. Unzip it, e.g. to `C:\llama.cpp\`.
3. Add that folder to your **PATH** (so `llama-server.exe` is found):

   ```powershell
   # Current session only:
   $env:Path = "C:\llama.cpp;" + $env:Path
   # Permanent (new terminals):
   setx PATH "C:\llama.cpp;$env:Path"
   ```

> On Windows, Surya runs on **CPU / CUDA / Vulkan** (there's no Metal). It works,
> but is slower than on Apple Silicon. **If you skip this**, the app still runs —
> the "Surya OCR" PDF modes just don't appear. You can instead use **native PDF
> text**, an **LM Studio** vision model, or upload **already-OCR'd `.txt`** files.

### 4. Run

```powershell
.\run.ps1                       # opens http://localhost:8501
# .\run.ps1 --server.port 8502  # use another port
```

If PowerShell blocks the script ("running scripts is disabled"):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or launch Streamlit directly, without the script:

```powershell
.\.venv-surya\Scripts\python -m streamlit run app.py
```

---

## Set up the LLM (the extraction model)

The app needs a model to extract from. Pick a provider in the **sidebar**.

### Option 1 — LM Studio (recommended local, macOS + Windows)

1. Install **LM Studio**: <https://lmstudio.ai>
2. In LM Studio, **search & download** a model. Good local picks for extraction:
   - macOS: **`Qwen3-30B-A3B-Instruct-2507` (MLX 4-bit or 6-bit)**, or **`Mistral-Small-3.2-24B-Instruct`**.
   - Windows: the **GGUF** builds of the same models (e.g. `Q4_K_M`).
3. Open the **Developer / Local Server** tab → **Start Server**
   (defaults to `http://localhost:1234`). Keep the model **loaded** (turn off
   idle auto-unload to avoid a cold-start error on the first request).
4. In the app sidebar: choose **"Local — LM Studio"**, click **Fetch models**,
   select your loaded model, run an extraction.

### Option 2 — Ollama (local)

```bash
# macOS
brew install ollama
# Windows: install from https://ollama.com/download

ollama pull qwen3            # or another tag
ollama serve                 # serves http://localhost:11434
```

Sidebar → **"Local — Ollama"** → fetch models → select.

### Option 3 — Cloud (OpenAI / Anthropic)

Sidebar → **"OpenAI (ChatGPT)"** or **"Anthropic (Claude)"** → paste your API key.

> ⚠️ **Sending reports to a cloud API transmits PHI off the machine.** For
> patient data, prefer a **local** model, or ensure you have the right
> agreements (HIPAA **BAA** / GDPR **DPA**) in place first. See the PHI note.

---

## Using the app (first run)

1. Pick a provider + model in the sidebar (above).
2. Keep the sample **instructions** (`config/instructions.txt`) and **schema**
   (`config/schema.json`), or select another preset / upload your own.
3. Paste a report, or upload one or more **`.txt` / `.pdf`** files (or reuse a
   saved **OCR `.json`** to skip OCR).
4. Run extraction → download the **CSV** and the **ZIP** of outputs.

The extra pages (top-left menu) — **Schema builder**, **Instructions manager**,
**Evaluation** — help you build schemas and benchmark models on your own reports.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit` runs the wrong Python / missing deps | Always launch via `./run.sh` (mac) or `.\run.ps1` (win) — they force the project venv. |
| Install fails with an `opencv` / Surya wheel error | You're on **Python 3.14**. Recreate the venv with **3.12**. |
| "Surya OCR: NOT ready" | Surya not installed or `llama-server` not on PATH. Mac: `brew install llama.cpp`. Win: add `llama-server.exe` to PATH (Part B §3). |
| Port 8501 already in use | `./run.sh --server.port 8502` (or `.\run.ps1 --server.port 8502`). |
| LM Studio: `Channel Error` on the first call | Cold start — the model was still loading. Turn off LM Studio's **auto-unload** so it stays warm; it auto-retries and succeeds. |
| Windows: "running scripts is disabled" | `powershell -ExecutionPolicy Bypass -File .\run.ps1`. |

**Live resource monitors** (`app_usage.sh`, `surya_usage.sh`) are **macOS-only**
(they use `vm_stat` / `sysctl` / `lsof`). There is no Windows equivalent — use
Task Manager, or LM Studio's own stats.

---

## PHI / privacy note

- With **Surya OCR + a local LLM** (LM Studio / Ollama), reports are processed
  **fully on the laptop** — nothing is sent to any server, no API agreement
  needed. This is the recommended setup for patient data.
- `ocr_cache/`, `.llm_usage.json`, and `.ocr_usage.json` are **git-ignored** —
  `ocr_cache/` holds report text (treat as PHI); never commit it.
- The Streamlit theme uses **built-in fonts only** (no web fonts) — nothing is
  fetched off-device at runtime.
- **Cloud** providers (OpenAI / Anthropic) transmit report text off the machine.
  Only use them for PHI with the appropriate **BAA (HIPAA)** / **DPA (GDPR)** in
  place.
