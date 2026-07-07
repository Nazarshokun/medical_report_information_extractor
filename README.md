# medical_report_information_extractor

Separate Streamlit project that replicates the core application approach described in:

`Leveraging large language models for structured information extraction from pathology reports`

This project does not modify `copression_pdf` or the earlier pathology app.

## What it replicates

- plaintext pathology reports as input
- PDF reports as input
- OpenAI-compatible API endpoint configuration
- model discovery through the `models` endpoint
- task behavior controlled by external configuration files
- JSON Schema-driven structured extraction

## Model providers

Pick a provider in the sidebar:

- **OpenAI (ChatGPT)** — the official OpenAI API (`https://api.openai.com/v1`).
- **Anthropic (Claude)** — the native Claude API via the `anthropic` SDK. Defaults
  to `claude-opus-4-8`. Requires an output token cap (set in the sidebar); the
  `temperature` and `Use JSON mode` controls do not apply to this backend.
- **Local — Ollama** — a local [Ollama](https://ollama.com) server
  (`http://localhost:11434/v1`). No API key needed.
- **Local — LM Studio** — a local LM Studio server (`http://localhost:1234/v1`).
- **Custom (OpenAI-compatible)** — any other OpenAI-compatible server, e.g.
  vLLM or llama.cpp. Fill in the base URL.

Local and custom OpenAI-compatible servers use the same code path as OpenAI;
only the base URL changes. Switching providers pre-fills a sensible base URL and
default model, both of which remain editable.

## PDF handling

- Word-generated or other born-digital PDFs can be processed with native PDF text extraction.
- Scanned PDFs can be processed through OCR fallback using `ocrmypdf` and `tesseract`.
- You can choose among:
  - native text only
  - auto mode: native text first, OCR fallback when little or no text is found
  - force OCR on all PDFs

## What it does not replicate

- the paper's original OCR + layout reconstruction pipeline
- de-identification pipeline
- gold-standard evaluation workflow from the paper

This app assumes you already have de-identified reports, whether as plaintext files or PDFs.

## Project files

- `app.py`: Streamlit UI
- `config/instructions.txt`: sample zero-shot extraction instructions
- `config/schema.json`: sample extraction schema

## Run

> **Deploying on a laptop (macOS or Windows)?** See **[DEPLOYMENT.md](DEPLOYMENT.md)** for
> the complete setup (Python 3.12 `.venv-surya`, on-device Surya OCR, LM Studio / Ollama).
> Once set up, launch with **`./run.sh`** (macOS) or **`.\run.ps1`** (Windows) — these use
> the project venv, which a bare `streamlit run` can miss.

```bash
cd medical_report_information_extractor
streamlit run app.py
```

If `streamlit` is not on your shell `PATH`, use the Python interpreter from your existing virtualenv:

```bash
/path/to/your/venv/bin/python -m streamlit run app.py
```

## Usage

1. Enter an OpenAI-compatible API base URL and API key.
2. Fetch models or type a model name manually.
3. Keep the sample instructions/schema or replace them with your own files.
4. Paste a plaintext report or upload one or more `.txt` and/or `.pdf` files.
5. Run extraction and download the CSV file and ZIP bundle of outputs.

## Notes

- The app validates the model output against the supplied JSON Schema and reports mismatches.
- Some OpenAI-compatible servers do not support JSON mode. Disable `Use JSON mode` if needed.
- PDF OCR support requires the local `ocrmypdf` and `tesseract` commands.
- CSV export includes `source_file_name` plus the schema keys as column headers, one row per successfully extracted report.
- The ZIP output includes the prepared plaintext source used for each report as `*.source.txt`.
