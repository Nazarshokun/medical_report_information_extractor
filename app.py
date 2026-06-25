from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import fitz
import streamlit as st
from jsonschema import Draft202012Validator
from openai import OpenAI

try:
    # Optional: only needed for the Anthropic (Claude) backend. The app still
    # runs for OpenAI-compatible and local providers when this is missing.
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

APP_TITLE = "Medical Report Information Extractor"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
MULTI_REPORT_CUE_PATTERNS = (
    r"(?im)^\s*(?:pathology report|surgical pathology report|anatomic pathology report)\b",
    r"(?im)^\s*(?:informe(?:\s+de)?\s+anatom[ií]a\s+patol[oó]gica|informe\s+de\s+biopsia)\b",
    r"(?im)^\s*(?:final diagnosis|diagnosis|diagn[oó]stico)\b",
    r"(?im)^\s*(?:patient name|patient|paciente|nombre)\b",
)

# Provider presets. The `openai` backend covers OpenAI cloud, local servers
# (Ollama, LM Studio, vLLM, llama.cpp), and any OpenAI-compatible endpoint.
# The `anthropic` backend uses the native Claude API.
DEFAULT_PROVIDER = "OpenAI (ChatGPT)"
PROVIDER_PRESETS: dict[str, dict] = {
    "OpenAI (ChatGPT)": {
        "backend": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.4",
        "needs_key": True,
    },
    "Anthropic (Claude)": {
        "backend": "anthropic",
        "base_url": "",
        "default_model": "claude-opus-4-8",
        "needs_key": True,
    },
    "Local — Ollama": {
        "backend": "openai",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "needs_key": False,
    },
    "Local — LM Studio": {
        "backend": "openai",
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
        "needs_key": False,
    },
    "Custom (OpenAI-compatible)": {
        "backend": "openai",
        "base_url": "",
        "default_model": "",
        "needs_key": False,
    },
}


@dataclass
class PreparedReport:
    report_name: str
    source_file_name: str
    report_text: str
    source_kind: str
    text_extraction_method: str
    preparation_warnings: list[str]


@dataclass
class ExtractionResult:
    report_name: str
    source_file_name: str
    success: bool
    status_label: str
    raw_response: str
    parsed_json: dict | list | None
    validation_errors: list[str]
    source_kind: str
    text_extraction_method: str
    preparation_warnings: list[str]
    prepared_text: str
    error_message: str | None = None


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(load_text(path))


def strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_response(text: str) -> dict | list:
    return json.loads(strip_code_fences(text))


def validate_schema(schema: dict) -> None:
    Draft202012Validator.check_schema(schema)


def validate_instance(instance: dict | list, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda err: list(err.path))
    messages: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{path}: " if path else ""
        messages.append(f"{prefix}{error.message}")
    return messages


def scalar_to_csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


EXTRA_MARKER_PREFIX = "extra__"


def collect_additional_marker_names(results: list[ExtractionResult]) -> list[str]:
    """Union of marker names found in `additional_markers` across all results.

    Returned sorted so the dynamic columns are stable for a given batch.
    """
    names: set[str] = set()
    for result in results:
        if not result.success or not isinstance(result.parsed_json, dict):
            continue
        extras = result.parsed_json.get("additional_markers")
        if not isinstance(extras, list):
            continue
        for entry in extras:
            if isinstance(entry, dict):
                name = entry.get("marker")
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
    return sorted(names)


def build_results_csv(results: list[ExtractionResult], schema: dict) -> bytes:
    # The `additional_markers` array is expanded into one `extra__<marker>`
    # column per unique marker found in the batch, instead of being dumped as a
    # single JSON cell. Every other schema field maps to its own column as before.
    base_fields = [
        field
        for field in schema.get("properties", {}).keys()
        if field != "additional_markers"
    ]
    extra_names = collect_additional_marker_names(results)
    fieldnames = [
        "source_file_name",
        *base_fields,
        *[EXTRA_MARKER_PREFIX + name for name in extra_names],
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for result in results:
        if not result.success or not isinstance(result.parsed_json, dict):
            continue
        row = {"source_file_name": result.source_file_name}
        row.update(
            {
                field: scalar_to_csv_cell(result.parsed_json.get(field))
                for field in base_fields
            }
        )
        extras = result.parsed_json.get("additional_markers")
        if isinstance(extras, list):
            for entry in extras:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("marker")
                if isinstance(name, str) and name.strip():
                    row[EXTRA_MARKER_PREFIX + name.strip()] = scalar_to_csv_cell(
                        entry.get("value")
                    )
        writer.writerow(row)

    return buffer.getvalue().encode("utf-8")


def build_results_zip(
    results: list[ExtractionResult],
    results_csv: bytes | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        summary = []
        for result in results:
            summary.append(
                {
                    "report_name": result.report_name,
                    "source_file_name": result.source_file_name,
                    "success": result.success,
                    "status": result.status_label,
                    "source_kind": result.source_kind,
                    "text_extraction_method": result.text_extraction_method,
                    "preparation_warnings": result.preparation_warnings,
                    "validation_errors": result.validation_errors,
                    "error_message": result.error_message,
                }
            )

            stem = Path(result.report_name).stem or "report"
            safe_stem = stem.replace("/", "_")
            zf.writestr(f"{safe_stem}.source.txt", result.prepared_text)
            if result.parsed_json is not None:
                zf.writestr(
                    f"{safe_stem}.json",
                    json.dumps(result.parsed_json, indent=2, ensure_ascii=False),
                )
            if result.error_message:
                zf.writestr(f"{safe_stem}.error.txt", result.error_message)

        if results_csv is not None:
            zf.writestr("results.csv", results_csv)
        zf.writestr("summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return buffer.getvalue()


@st.cache_resource(show_spinner=False)
def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    # Local servers often need no real key; pass a placeholder so the SDK
    # does not raise on an empty value. Blank base_url falls back to the
    # OpenAI default endpoint.
    return OpenAI(
        base_url=(base_url or "").rstrip("/") or None,
        api_key=api_key or "not-needed",
    )


@st.cache_resource(show_spinner=False)
def get_anthropic_client(base_url: str, api_key: str):
    if Anthropic is None:
        raise RuntimeError("The `anthropic` package is not installed. Run: pip install anthropic")
    if (base_url or "").strip():
        return Anthropic(api_key=api_key, base_url=base_url.strip())
    return Anthropic(api_key=api_key)


def fetch_models(client) -> list[str]:
    return sorted(model.id for model in client.models.list().data)


def run_chat_json(
    *,
    backend: str,
    client,
    model: str,
    system: str,
    user: str,
    use_json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> str:
    """Send one prompt and return the raw model text (expected to be JSON).

    Both backends receive an identical (system, user) split so prompts and
    downstream JSON parsing stay uniform across providers.
    """
    if backend == "anthropic":
        # The Anthropic Messages API takes the system prompt as a top-level
        # argument and requires max_tokens. Temperature and OpenAI-style JSON
        # mode do not apply (and temperature 400s on current Claude models).
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if getattr(message, "stop_reason", None) == "refusal":
            raise RuntimeError("The model declined to respond to this request (safety refusal).")
        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )

    request_kwargs = {
        "model": model,
        "temperature": temperature,
        # Bound the output so a runaway/repetition loop fails fast instead of
        # generating until it fills the whole context window. OpenAI-compatible
        # servers (incl. Ollama) map this to the generation limit.
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if use_json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**request_kwargs)
    return completion.choices[0].message.content or ""


TRIAGE_SYSTEM = (
    "You are a fast document classifier. You receive the raw text of a document "
    "and must decide whether it is an IMMUNOHISTOCHEMISTRY pathology report, i.e. "
    "a report that actually lists immunohistochemistry marker results (markers "
    "such as CD3, CD20, CD30, CD45, Ki-67, ALK, BCL6, EMA, TdT, etc., usually "
    "reported as positive/negative/percentage). Respond with EXACTLY one of these "
    "words and nothing else:\n"
    "- pathology : it IS a report that contains immunohistochemistry marker results\n"
    "- flow_citometry : it is a flow cytometry report (marker results by flow, not IHC)\n"
    "- not_report : anything else, INCLUDING pathology or medical reports that do "
    "NOT contain immunohistochemistry marker results (e.g. gross descriptions, "
    "clinical notes, molecular-only reports, cover letters)\n"
    "Only answer pathology if you can see actual immunohistochemistry marker "
    "results in the text. If genuinely unsure, answer pathology."
)


def triage_document(
    *,
    backend: str,
    client,
    model: str,
    report_text: str,
    temperature: float,
) -> str:
    """Cheap pre-screen: classify a document before the expensive extraction.

    Returns one of "pathology", "flow_citometry", "not_report". Only a tiny
    output is requested, and only the first ~6000 characters are sent, so this
    is far cheaper than a full schema extraction. The caller skips the heavy
    extraction whenever the result is not "pathology".
    """
    user = "Document text:\n\n" + report_text[:6000] + "\n\nClassification:"
    raw = run_chat_json(
        backend=backend,
        client=client,
        model=model,
        system=TRIAGE_SYSTEM,
        user=user,
        use_json_mode=False,
        temperature=temperature,
        max_tokens=8,
    )
    label = raw.strip().lower()
    if "flow" in label or "citometr" in label or "cytometr" in label:
        return "flow_citometry"
    if label.startswith("not") or "not_report" in label or "not a" in label:
        return "not_report"
    # Default to processing the document so a real report is never silently dropped.
    return "pathology"


def ihc_marker_keywords(schema: dict) -> set[str]:
    """Lower-cased marker name stems taken from the schema's coded marker fields.

    Used as a cheap keyword backstop: any integer/enum field is treated as an
    immunohistochemistry marker, and its core token (before a parenthesis or
    slash) becomes a search keyword, e.g. "CD56 (NCAM)" -> "cd56".
    """
    keywords: set[str] = set()
    for name, spec in schema.get("properties", {}).items():
        types = spec.get("type")
        is_marker = isinstance(types, list) and "integer" in types and "enum" in spec
        if not is_marker:
            continue
        core = re.split(r"[(/]", name)[0].strip().lower()
        if core:
            keywords.add(core)
    return keywords


def has_ihc_signal(text: str, marker_keywords: set[str]) -> bool:
    """True if the text shows any immunohistochemistry signal.

    Looks for an "immunohistochemistry" term (EN/PT/ES) or any known marker
    name. Used to skip documents that contain no IHC content at all without
    spending an LLM triage call on them.
    """
    low = text.lower()
    if any(term in low for term in ("immunohist", "imunohist", "inmunohist")):
        return True
    return any(keyword in low for keyword in marker_keywords)


@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_texts: list[str] = []
        total_chars = 0
        for page_index in range(doc.page_count):
            page_text = (doc.load_page(page_index).get_text("text") or "").strip()
            if page_text:
                page_texts.append(f"[Page {page_index + 1}]\n{page_text}")
                total_chars += len(page_text)
        return "\n\n".join(page_texts).strip(), total_chars
    finally:
        doc.close()


def assess_text_quality(report_text: str) -> tuple[float, list[str]]:
    lines = [
        line.strip()
        for line in (report_text or "").splitlines()
        if line.strip() and not re.fullmatch(r"\[Page \d+\]", line.strip())
    ]
    if not lines:
        return -1000.0, ["no extracted text"]

    colon_only_count = sum(1 for line in lines if re.fullmatch(r"[:;|]+", line))
    colon_prefixed_count = sum(1 for line in lines if line.startswith(":"))
    short_line_count = sum(1 for line in lines if len(line) <= 2)
    alpha_chars = sum(sum(char.isalpha() for char in line) for line in lines)
    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)

    reasons: list[str] = []
    colon_only_ratio = colon_only_count / max(len(lines), 1)
    short_line_ratio = short_line_count / max(len(lines), 1)
    if colon_only_count >= 4 or colon_only_ratio >= 0.12:
        reasons.append("too many standalone separator lines")
    if colon_prefixed_count >= 4:
        reasons.append("too many colon-prefixed fragments")
    if short_line_ratio >= 0.22 and len(lines) >= 20:
        reasons.append("too many very short lines")
    if avg_line_length < 18 and len(lines) >= 20:
        reasons.append("line layout looks fragmented")

    score = (alpha_chars / max(len(lines), 1)) + (avg_line_length * 0.5)
    score -= colon_only_count * 18
    score -= colon_prefixed_count * 8
    score -= max(short_line_count - 2, 0) * 2
    return score, reasons


def run_ocrmypdf_and_extract_text(
    *,
    pdf_bytes: bytes,
    ocrmypdf_path: str,
    ocr_languages: str,
    force_ocr: bool,
) -> tuple[str, list[str]]:
    with tempfile.TemporaryDirectory(prefix="mrie_pdf_") as temp_dir:
        input_path = Path(temp_dir) / "input.pdf"
        output_path = Path(temp_dir) / "output.pdf"
        sidecar_path = Path(temp_dir) / "output.txt"
        input_path.write_bytes(pdf_bytes)

        command = [
            ocrmypdf_path,
            "--output-type",
            "pdf",
            "--tagged-pdf-mode",
            "ignore",
            "--sidecar",
            str(sidecar_path),
        ]

        if ocr_languages.strip():
            command.extend(["-l", ocr_languages.strip()])

        command.extend(["--mode", "force" if force_ocr else "skip"])
        command.extend([str(input_path), str(output_path)])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        logs = [
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        ]
        if completed.returncode != 0 or not output_path.exists():
            log_text = "\n\n".join(logs) if logs else f"ocrmypdf failed with exit code {completed.returncode}"
            raise RuntimeError(log_text)

        ocr_text, _ = extract_text_from_pdf_bytes(output_path.read_bytes())
        if not ocr_text.strip() and sidecar_path.exists():
            ocr_text = sidecar_path.read_text(encoding="utf-8", errors="replace").strip()

        return ocr_text, logs


def prepare_pdf_report(
    *,
    report_name: str,
    pdf_bytes: bytes,
    pdf_input_mode: str,
    native_text_min_chars: int,
    ocr_languages: str,
    ocrmypdf_path: str | None,
) -> PreparedReport:
    native_text, native_chars = extract_text_from_pdf_bytes(pdf_bytes)
    warnings: list[str] = []
    native_score, native_quality_reasons = assess_text_quality(native_text)

    if pdf_input_mode == "native":
        if not native_text.strip():
            warnings.append("No native PDF text was found.")
        elif native_quality_reasons:
            warnings.append(
                "Native PDF text may be low quality: " + "; ".join(native_quality_reasons) + "."
            )
        return PreparedReport(
            report_name=report_name,
            source_file_name=report_name,
            report_text=native_text,
            source_kind="pdf",
            text_extraction_method="pdf-native",
            preparation_warnings=warnings,
        )

    should_run_ocr = pdf_input_mode == "force_ocr"
    if pdf_input_mode == "auto_ocr_fallback":
        if native_chars < native_text_min_chars:
            should_run_ocr = True
            warnings.append(
                f"Native PDF text was short ({native_chars} chars), so OCR was attempted."
            )
        if native_quality_reasons:
            should_run_ocr = True
            warnings.append(
                "Native PDF text looked low quality, so OCR was attempted: "
                + "; ".join(native_quality_reasons)
                + "."
            )

    if should_run_ocr:
        if not ocrmypdf_path:
            raise RuntimeError("ocrmypdf is not available, so OCR cannot be used for this PDF.")

        force_ocr_for_attempt = pdf_input_mode == "force_ocr" or (
            pdf_input_mode == "auto_ocr_fallback" and bool(native_text.strip())
        )
        ocr_text, ocr_logs = run_ocrmypdf_and_extract_text(
            pdf_bytes=pdf_bytes,
            ocrmypdf_path=ocrmypdf_path,
            ocr_languages=ocr_languages,
            force_ocr=force_ocr_for_attempt,
        )
        ocr_score, _ = assess_text_quality(ocr_text)
        if ocr_logs:
            warnings.append("OCRmyPDF was run for this report.")

        if ocr_text.strip():
            if pdf_input_mode != "force_ocr" and native_text.strip() and ocr_score + 5 < native_score:
                warnings.append(
                    "OCR text did not look better than native PDF text, so native text was kept."
                )
                return PreparedReport(
                    report_name=report_name,
                    source_file_name=report_name,
                    report_text=native_text,
                    source_kind="pdf",
                    text_extraction_method="pdf-native",
                    preparation_warnings=warnings,
                )

            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=ocr_text,
                source_kind="pdf",
                text_extraction_method="pdf-ocr-forced" if pdf_input_mode == "force_ocr" else "pdf-ocr-auto",
                preparation_warnings=warnings,
            )

        if native_text.strip():
            warnings.append("OCR produced no usable text, so native PDF text was used instead.")
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=native_text,
                source_kind="pdf",
                text_extraction_method="pdf-native",
                preparation_warnings=warnings,
            )

        raise RuntimeError("No text could be extracted from this PDF.")

    return PreparedReport(
        report_name=report_name,
        source_file_name=report_name,
        report_text=native_text,
        source_kind="pdf",
        text_extraction_method="pdf-native",
        preparation_warnings=warnings,
    )


def should_attempt_multi_report_split(report_text: str) -> bool:
    text = (report_text or "").strip()
    if not text:
        return False
    cue_hits = sum(len(re.findall(pattern, text)) for pattern in MULTI_REPORT_CUE_PATTERNS)
    if cue_hits >= 3:
        return True
    if len(re.findall(r"(?im)^\[Page \d+\]", text)) >= 2 and cue_hits >= 2:
        return True
    return False


def build_segment_report_name(source_file_name: str, index: int, total: int) -> str:
    if total <= 1:
        return source_file_name
    source_path = Path(source_file_name)
    stem = source_path.stem or "report"
    suffix = source_path.suffix or ".txt"
    return f"{stem}__report_{index}{suffix}"


def split_prepared_report(
    *,
    backend: str,
    client,
    model: str,
    prepared_report: PreparedReport,
    use_json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> list[PreparedReport]:
    if not should_attempt_multi_report_split(prepared_report.report_text):
        return [prepared_report]

    prompt = f"""
Split this source document into distinct pathology reports if the file contains more than one report.

Return exactly one JSON object with this shape:
{{
  "reports": [
    {{
      "label": "report_1",
      "report_text": "exact text copied from the source for one report"
    }}
  ]
}}

Rules:
- Keep the original order.
- Copy the report text from the source as exactly as possible.
- Do not translate, summarize, clean up, or normalize the text.
- Do not invent report boundaries.
- If the source contains only one report, return one item with the full original text.
- Each `report_text` must be a self-contained report segment.

Original source filename:
{prepared_report.source_file_name}

Source text:
{prepared_report.report_text}
""".strip()

    try:
        content = run_chat_json(
            backend=backend,
            client=client,
            model=model,
            system=(
                "You split source documents into distinct pathology reports. "
                "Return JSON only."
            ),
            user=prompt,
            use_json_mode=use_json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        ) or "{}"
        parsed = parse_json_response(content)
        candidate_reports = parsed.get("reports") if isinstance(parsed, dict) else None
        if not isinstance(candidate_reports, list):
            raise ValueError("Splitter response did not contain a `reports` list.")

        segment_texts: list[str] = []
        for item in candidate_reports:
            if not isinstance(item, dict):
                continue
            segment_text = str(item.get("report_text") or "").strip()
            if segment_text:
                segment_texts.append(segment_text)

        if not segment_texts:
            raise ValueError("Splitter returned no non-empty report segments.")

        original_length = max(len(prepared_report.report_text.strip()), 1)
        combined_length = sum(len(segment_text) for segment_text in segment_texts)
        if len(segment_texts) == 1 and combined_length < original_length * 0.5:
            raise ValueError("Splitter returned a suspiciously short single segment.")
        if len(segment_texts) > 1 and combined_length < original_length * 0.5:
            raise ValueError("Splitter returned suspiciously short combined segments.")

        if len(segment_texts) == 1:
            return [prepared_report]

        split_warning = f"Auto-split this file into {len(segment_texts)} report segments."
        split_reports: list[PreparedReport] = []
        for index, segment_text in enumerate(segment_texts, start=1):
            split_reports.append(
                PreparedReport(
                    report_name=build_segment_report_name(
                        prepared_report.source_file_name,
                        index,
                        len(segment_texts),
                    ),
                    source_file_name=prepared_report.source_file_name,
                    report_text=segment_text,
                    source_kind=prepared_report.source_kind,
                    text_extraction_method=f"{prepared_report.text_extraction_method}+multi-report-split",
                    preparation_warnings=[*prepared_report.preparation_warnings, split_warning],
                )
            )
        return split_reports
    except Exception as exc:
        fallback_warning = (
            "Multi-report splitting was attempted but failed, so the full file was extracted as one report. "
            f"Reason: {exc}"
        )
        return [
            PreparedReport(
                report_name=prepared_report.report_name,
                source_file_name=prepared_report.source_file_name,
                report_text=prepared_report.report_text,
                source_kind=prepared_report.source_kind,
                text_extraction_method=prepared_report.text_extraction_method,
                preparation_warnings=[*prepared_report.preparation_warnings, fallback_warning],
            )
        ]


def extract_reports(
    *,
    backend: str,
    client,
    model: str,
    instructions: str,
    schema: dict,
    prepared_reports: list[PreparedReport],
    use_json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []

    for prepared_report in prepared_reports:
        report_name = prepared_report.report_name
        report_text = prepared_report.report_text
        prompt = f"""
Extract structured information from this pathology report.

Return exactly one JSON object that follows this JSON Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Report/source filename:
{prepared_report.source_file_name}

Report segment label:
{report_name}

Pathology report text:
{report_text}
""".strip()

        try:
            content = run_chat_json(
                backend=backend,
                client=client,
                model=model,
                system=instructions,
                user=prompt,
                use_json_mode=use_json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            ) or "{}"
            parsed = parse_json_response(content)
            validation_errors = validate_instance(parsed, schema)
            results.append(
                ExtractionResult(
                    report_name=report_name,
                    source_file_name=prepared_report.source_file_name,
                    success=True,
                    status_label="valid" if not validation_errors else "schema-warning",
                    raw_response=content,
                    parsed_json=parsed,
                    validation_errors=validation_errors,
                    source_kind=prepared_report.source_kind,
                    text_extraction_method=prepared_report.text_extraction_method,
                    preparation_warnings=prepared_report.preparation_warnings,
                    prepared_text=report_text,
                )
            )
        except Exception as exc:
            results.append(
                ExtractionResult(
                    report_name=report_name,
                    source_file_name=prepared_report.source_file_name,
                    success=False,
                    status_label="failed",
                    raw_response="",
                    parsed_json=None,
                    validation_errors=[],
                    source_kind=prepared_report.source_kind,
                    text_extraction_method=prepared_report.text_extraction_method,
                    preparation_warnings=prepared_report.preparation_warnings,
                    prepared_text=report_text,
                    error_message=str(exc),
                )
            )

    return results


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.write(
    "A separate Streamlit project that mirrors the paper's approach: report text "
    "or PDFs in, a selectable model endpoint (OpenAI, a local LLM, or Anthropic "
    "Claude), external instructions and JSON Schema, structured JSON out."
)
st.caption(
    "This app accepts plaintext reports or PDFs. Word-generated PDFs can use native "
    "text extraction, while scanned PDFs can use OCR. De-identification is still outside this project."
)

default_instructions = load_text(CONFIG_DIR / "instructions.txt")
default_schema = load_json(CONFIG_DIR / "schema.json")

st.session_state.setdefault("provider", DEFAULT_PROVIDER)
st.session_state.setdefault("available_models", [])
if "base_url" not in st.session_state:
    st.session_state["base_url"] = PROVIDER_PRESETS[st.session_state["provider"]]["base_url"]
if "model_text" not in st.session_state:
    st.session_state["model_text"] = PROVIDER_PRESETS[st.session_state["provider"]]["default_model"]


def _on_provider_change() -> None:
    chosen = PROVIDER_PRESETS[st.session_state["provider"]]
    st.session_state["base_url"] = chosen["base_url"]
    st.session_state["model_text"] = chosen["default_model"]
    st.session_state["available_models"] = []


with st.sidebar:
    st.header("Model Endpoint")
    provider_name = st.selectbox(
        "Provider",
        list(PROVIDER_PRESETS.keys()),
        key="provider",
        on_change=_on_provider_change,
        help=(
            "OpenAI and local servers (Ollama, LM Studio, vLLM) use the OpenAI-compatible "
            "API. Anthropic uses the native Claude API."
        ),
    )
    preset = PROVIDER_PRESETS[provider_name]
    backend = preset["backend"]

    base_url = st.text_input(
        "Base URL",
        key="base_url",
        help="OpenAI-compatible base URL. Leave blank to use the Anthropic default.",
    )
    api_key = st.text_input(
        "API key",
        type="password",
        key="api_key",
        help="Required for OpenAI and Anthropic. Usually optional for local servers.",
    )

    if backend == "anthropic" and not ANTHROPIC_AVAILABLE:
        st.warning("Install the Anthropic SDK to use Claude: `pip install anthropic`")

    if st.button("Fetch models"):
        if preset["needs_key"] and not api_key.strip():
            st.error("API key is required to fetch models for this provider.")
        elif backend == "anthropic" and not ANTHROPIC_AVAILABLE:
            st.error("The `anthropic` package is not installed.")
        else:
            try:
                if backend == "anthropic":
                    fetch_client = get_anthropic_client(base_url, api_key)
                else:
                    fetch_client = get_openai_client(base_url, api_key)
                st.session_state["available_models"] = fetch_models(fetch_client)
                st.success(f"Loaded {len(st.session_state['available_models'])} model(s).")
            except Exception as exc:
                st.error(f"Could not fetch models: {exc}")

    available_models: list[str] = st.session_state["available_models"]
    if available_models:
        model = st.selectbox("Model", available_models, key="model_select")
    else:
        model = st.text_input("Model", key="model_text")

    if backend == "anthropic":
        anthropic_max_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=16000,
            value=8000,
            step=256,
            help="Anthropic requires an output token cap. Raise it for very long reports.",
        )
        use_json_mode = False
        temperature = 0.0
        st.caption("Temperature and JSON mode do not apply to the Anthropic backend.")
    else:
        use_json_mode = st.checkbox(
            "Use JSON mode",
            value=True,
            help="Disable this if your OpenAI-compatible server does not support `response_format=json_object`.",
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        anthropic_max_tokens = 8000

    split_multi_report_files = st.checkbox(
        "Split files with multiple reports",
        value=True,
        help="If one uploaded file contains several pathology reports, try to split it into separate extractions.",
    )
    prescreen_skip = st.checkbox(
        "Pre-screen and skip non-pathology documents",
        value=True,
        help=(
            "Run a fast one-word classification on each document first. Documents "
            "that are not pathology/immunohistochemistry reports are marked "
            "not_report (or flow_citometry) and the slow full extraction is "
            "skipped, saving time. If the screen is unsure it still runs the full "
            "extraction, so real reports are never dropped."
        ),
    )
    st.header("PDF Input")
    pdf_input_mode = st.selectbox(
        "PDF text preparation",
        ["auto_ocr_fallback", "native", "force_ocr"],
        index=0,
        format_func=lambda value: {
            "auto_ocr_fallback": "Auto: native text, OCR if short or low quality",
            "native": "Native text only",
            "force_ocr": "Force OCR on all PDFs",
        }[value],
    )
    ocr_languages = st.text_input(
        "OCR language(s)",
        value="eng",
        help="Tesseract language codes joined with `+`, for example `eng+spa+por`.",
    )
    native_text_min_chars = st.number_input(
        "Min native PDF chars before OCR",
        min_value=0,
        max_value=10000,
        value=80,
        step=20,
        help="In auto mode, PDFs below this native-text threshold or with poor native layout are sent through OCR.",
    )

config_col, report_col = st.columns([1, 1])

with config_col:
    st.subheader("Configuration")

    instructions_file = st.file_uploader("Instructions file (.txt)", type=["txt"])
    instructions_text = (
        instructions_file.getvalue().decode("utf-8", errors="replace")
        if instructions_file is not None
        else default_instructions
    )
    instructions_text = st.text_area(
        "Task instructions",
        value=instructions_text,
        height=220,
    )

    schema_file = st.file_uploader("JSON Schema file (.json)", type=["json"])
    schema_text = (
        schema_file.getvalue().decode("utf-8", errors="replace")
        if schema_file is not None
        else json.dumps(default_schema, indent=2, ensure_ascii=False)
    )
    schema_text = st.text_area(
        "JSON Schema",
        value=schema_text,
        height=320,
    )

with report_col:
    st.subheader("Reports")
    pymupdf_available = True
    ocrmypdf_path = shutil.which("ocrmypdf")
    tesseract_path = shutil.which("tesseract")

    dep_cols = st.columns(3)
    dep_cols[0].metric("PyMuPDF", "Found" if pymupdf_available else "Missing")
    dep_cols[1].metric("ocrmypdf", "Found" if ocrmypdf_path else "Missing")
    dep_cols[2].metric("tesseract", "Found" if tesseract_path else "Missing")

    pasted_report = st.text_area(
        "Paste one plaintext pathology report",
        value="",
        height=220,
    )
    uploaded_reports = st.file_uploader(
        "Upload plaintext report files (.txt/.md)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    uploaded_pdf_reports = st.file_uploader(
        "Upload PDF report files (.pdf)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.info(
        "Replicated from the paper's architecture: the model receives extracted plaintext "
        "plus external instructions and a JSON Schema. PDFs from Word usually work through "
        "native text extraction, while scanned PDFs can go through OCR."
    )

run_extraction = st.button("Extract structured information", type="primary")

if run_extraction:
    if preset["needs_key"] and not api_key.strip():
        st.error("API key is required for this provider.")
        st.stop()
    if not model.strip():
        st.error("Model is required.")
        st.stop()
    if not instructions_text.strip():
        st.error("Task instructions are required.")
        st.stop()
    if backend == "anthropic" and not ANTHROPIC_AVAILABLE:
        st.error("The `anthropic` package is not installed. Run `pip install anthropic`.")
        st.stop()

    try:
        schema_obj = json.loads(schema_text)
        validate_schema(schema_obj)
    except Exception as exc:
        st.error(f"Invalid JSON Schema: {exc}")
        st.stop()

    try:
        if backend == "anthropic":
            llm_client = get_anthropic_client(base_url, api_key)
        else:
            llm_client = get_openai_client(base_url, api_key)
    except Exception as exc:
        st.error(f"Could not initialize the model client: {exc}")
        st.stop()

    prepared_reports: list[PreparedReport] = []
    if pasted_report.strip():
        prepared_reports.append(
            PreparedReport(
                report_name="pasted_report.txt",
                source_file_name="pasted_report.txt",
                report_text=pasted_report.strip(),
                source_kind="plain-text",
                text_extraction_method="direct-text",
                preparation_warnings=[],
            )
        )
    for uploaded_file in uploaded_reports or []:
        prepared_reports.append(
            PreparedReport(
                report_name=uploaded_file.name,
                source_file_name=uploaded_file.name,
                report_text=uploaded_file.getvalue().decode("utf-8", errors="replace"),
                source_kind="plain-text",
                text_extraction_method="uploaded-text",
                preparation_warnings=[],
            )
        )
    for uploaded_pdf in uploaded_pdf_reports or []:
        try:
            prepared_reports.append(
                prepare_pdf_report(
                    report_name=uploaded_pdf.name,
                    pdf_bytes=uploaded_pdf.getvalue(),
                    pdf_input_mode=pdf_input_mode,
                    native_text_min_chars=int(native_text_min_chars),
                    ocr_languages=ocr_languages,
                    ocrmypdf_path=ocrmypdf_path,
                )
            )
        except Exception as exc:
            prepared_reports.append(
                PreparedReport(
                    report_name=uploaded_pdf.name,
                    source_file_name=uploaded_pdf.name,
                    report_text="",
                    source_kind="pdf",
                    text_extraction_method="pdf-preparation-failed",
                    preparation_warnings=[str(exc)],
                )
            )

    if not prepared_reports:
        st.error("Provide at least one plaintext report or PDF.")
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()
    expanded_reports: list[PreparedReport] = []

    for index, prepared_report in enumerate(prepared_reports, start=1):
        status.write(f"Preparing {prepared_report.report_name} ({index}/{len(prepared_reports)})")
        if split_multi_report_files and prepared_report.report_text.strip():
            expanded_reports.extend(
                split_prepared_report(
                    backend=backend,
                    client=llm_client,
                    model=model,
                    prepared_report=prepared_report,
                    use_json_mode=use_json_mode,
                    temperature=float(temperature),
                    max_tokens=16000,
                )
            )
        else:
            expanded_reports.append(prepared_report)
        progress.progress(index / max(len(prepared_reports), 1))

    marker_keywords = ihc_marker_keywords(schema_obj)
    results: list[ExtractionResult] = []
    for index, prepared_report in enumerate(expanded_reports, start=1):
        status.write(f"Extracting {prepared_report.report_name} ({index}/{len(expanded_reports)})")
        if not prepared_report.report_text.strip():
            results.append(
                ExtractionResult(
                    report_name=prepared_report.report_name,
                    source_file_name=prepared_report.source_file_name,
                    success=False,
                    status_label="failed",
                    raw_response="",
                    parsed_json=None,
                    validation_errors=[],
                    source_kind=prepared_report.source_kind,
                    text_extraction_method=prepared_report.text_extraction_method,
                    preparation_warnings=prepared_report.preparation_warnings,
                    prepared_text=prepared_report.report_text,
                    error_message=prepared_report.preparation_warnings[0]
                    if prepared_report.preparation_warnings
                    else "No text could be prepared for this report.",
                )
            )
            progress.progress(index / max(len(expanded_reports), 1))
            continue

        if prescreen_skip:
            status.write(
                f"Screening {prepared_report.report_name} ({index}/{len(expanded_reports)})"
            )
            if not has_ihc_signal(prepared_report.report_text, marker_keywords):
                # Keyword backstop: no immunohistochemistry marker or term appears
                # anywhere in the text, so skip immediately without spending an
                # LLM triage call.
                triage_label = "not_report"
            else:
                try:
                    triage_label = triage_document(
                        backend=backend,
                        client=llm_client,
                        model=model,
                        report_text=prepared_report.report_text,
                        temperature=float(temperature),
                    )
                except Exception:
                    # If the screen fails, fall through to a full extraction rather
                    # than risk dropping a real report.
                    triage_label = "pathology"
            if triage_label != "pathology":
                results.append(
                    ExtractionResult(
                        report_name=prepared_report.report_name,
                        source_file_name=prepared_report.source_file_name,
                        success=True,
                        status_label=triage_label,
                        raw_response="(pre-screened; full extraction skipped)",
                        parsed_json={"not_report": triage_label},
                        validation_errors=[],
                        source_kind=prepared_report.source_kind,
                        text_extraction_method=prepared_report.text_extraction_method,
                        preparation_warnings=prepared_report.preparation_warnings,
                        prepared_text=prepared_report.report_text,
                        error_message=None,
                    )
                )
                progress.progress(index / max(len(expanded_reports), 1))
                continue

        single_result = extract_reports(
            backend=backend,
            client=llm_client,
            model=model,
            instructions=instructions_text,
            schema=schema_obj,
            prepared_reports=[prepared_report],
            use_json_mode=use_json_mode,
            temperature=float(temperature),
            max_tokens=int(anthropic_max_tokens),
        )[0]
        results.append(single_result)
        progress.progress(index / max(len(expanded_reports), 1))

    progress.empty()
    status.empty()

    st.subheader("Results")
    summary_rows = [
        {
            "report": result.report_name,
            "source_file": result.source_file_name,
            "source": result.source_kind,
            "text_mode": result.text_extraction_method,
            "status": result.status_label,
            "validation_errors": len(result.validation_errors),
        }
        for result in results
    ]
    st.dataframe(summary_rows, use_container_width=True)

    successful_csv_results = [
        result for result in results if result.success and isinstance(result.parsed_json, dict)
    ]
    results_csv = build_results_csv(successful_csv_results, schema_obj)
    bundle = build_results_zip(results, results_csv=results_csv)

    if successful_csv_results:
        st.download_button(
            "Download results (.csv)",
            data=results_csv,
            file_name="results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No successful structured rows were available for CSV export.")

    st.download_button(
        "Download results (.zip)",
        data=bundle,
        file_name="medical_report_information_extractor_results.zip",
        mime="application/zip",
    )

    for result in results:
        with st.expander(result.report_name, expanded=not result.success):
            st.write(f"Source filename: `{result.source_file_name}`")
            st.write(f"Source kind: `{result.source_kind}`")
            st.write(f"Text extraction method: `{result.text_extraction_method}`")
            if result.preparation_warnings:
                st.warning("\n".join(result.preparation_warnings))
            if result.error_message:
                st.error(result.error_message)
                continue

            if result.validation_errors:
                st.warning("Output parsed, but it does not fully match the schema.")
                st.code("\n".join(result.validation_errors), language="text")

            input_tab, json_tab, raw_tab = st.tabs(["Prepared text", "JSON", "Raw response"])
            with input_tab:
                st.code(result.prepared_text or "No prepared text captured.", language="text")
            with json_tab:
                st.json(result.parsed_json)
            with raw_tab:
                st.code(result.raw_response or "No raw response captured.", language="json")
