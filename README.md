# medical_report_information_extractor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23024522.svg)](https://doi.org/10.5281/zenodo.23024522)

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

## Patient data: local processing only

For registry work and any real patient documents, use only the local providers
(**Ollama**, **LM Studio**, or a self-hosted OpenAI-compatible server such as vLLM
or llama.cpp) together with on-device OCR. In this setup, report text, OCR and
model inference all stay on the hospital computer; nothing is sent to a
third-party or cloud service. The OpenAI and Anthropic options are intended only
for synthetic or already public test documents.

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

## Roadmap

- **Local de-identification** of reports before extraction, running on-premises
  like the rest of the pipeline, with a target recall of ≥99% for personal
  identifiers.
- **Discharge summaries to pre-fill registry registration forms** for data-manager
  verification. A first schema and instructions preset are included
  (`config/schema_discharge.json`, `config/instructions/discharge_summary.txt`);
  the fields follow the lymphoma registry REDCap variable names.
- **Better accuracy on scanned documents**, where extraction from poor scans is
  currently the main source of error.
- **Multicenter validation** against manual double abstraction in lymphoma
  registries in Ukraine, Moldova, Kazakhstan and Romania.

## Project files

- `app.py`: Streamlit UI
- `config/instructions.txt`: sample zero-shot extraction instructions
- `config/schema.json`: sample extraction schema
- `config/schema_discharge.json` and `config/instructions/discharge_summary.txt`:
  discharge-summary preset mapped to lymphoma registry REDCap fields
- `CITATION.cff` and `.zenodo.json`: citation metadata

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
- CSV export includes `source_file_name`, `extraction_status`, and the schema keys as column headers, one row per successfully extracted report. `extraction_status` is `valid`, `schema-warning`, `truncated`, or `needs-review` — or `not_report` / `flow_citometry` for documents the pre-screen skipped, whose fields are left blank.
- "Download for Excel" gives the same CSV with a UTF-8 byte-order mark so Excel shows accented and Cyrillic text correctly; the plain `results.csv` stays BOM-free for R / pandas.
- The ZIP output includes the prepared plaintext source used for each report as `*.source.txt`.

## Citation

If you use this software, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository").
Released versions are archived on Zenodo: [doi.org/10.5281/zenodo.23024522](https://doi.org/10.5281/zenodo.23024522)
(this DOI always resolves to the latest version).

## License

MIT — see [LICENSE](LICENSE).
