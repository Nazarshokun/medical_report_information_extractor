from __future__ import annotations

import base64
import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import fitz
import streamlit as st
from jsonschema import Draft202012Validator
from openai import BadRequestError, OpenAI, UnprocessableEntityError

try:
    # Optional: only needed for the Anthropic (Claude) backend. The app still
    # runs for OpenAI-compatible and local providers when this is missing.
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    # Optional: Apple Vision on-device OCR (macOS only). Pulls Pillow + pyobjc.
    from ocrmac import ocrmac as ocrmac_engine

    OCRMAC_AVAILABLE = True
except Exception:  # pragma: no cover - macOS-only, optional
    ocrmac_engine = None
    OCRMAC_AVAILABLE = False

try:
    # Optional: Datalab lift — on-device PDF/image -> schema JSON via the 9B
    # datalab-to/lift vision model (HuggingFace backend). torch/transformers are
    # imported lazily inside lift, so this import succeeds even without the [hf]
    # extra; the model load then raises a clear install hint if they're missing.
    from lift import extract as lift_extract, resolve_schema as lift_resolve_schema
    from lift.model import InferenceManager as LiftInferenceManager
    from lift.settings import settings as lift_settings

    LIFT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    lift_extract = lift_resolve_schema = LiftInferenceManager = lift_settings = None
    LIFT_AVAILABLE = False

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


@dataclass
class ExtractionOutcome:
    """Extraction result for one report, without per-file metadata.

    Cached on (report text + all model/schema/prompt settings); metadata such as
    report_name / source_kind / preparation_warnings is attached afterwards.
    """

    success: bool
    status_label: str
    raw_response: str
    parsed_json: dict | list | None
    validation_errors: list[str]
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


def _extract_json_blob(text: str) -> str:
    """Best-effort recovery of a JSON object/array from surrounding prose.

    Reasoning/"thinking" models (and runners that ignore JSON mode) often wrap
    the JSON in a <think> block or explanatory text. Strip those and return the
    substring from the first opening bracket to the last closing one.
    """
    without_think = re.sub(
        r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    start = next((i for i, ch in enumerate(without_think) if ch in "{["), None)
    if start is None:
        return without_think
    end = max(without_think.rfind("}"), without_think.rfind("]"))
    if end < start:
        return without_think
    return without_think[start : end + 1]


def _repair_truncated_json(text: str) -> str:
    """Best-effort completion of JSON that was truncated mid-generation.

    When a model loops inside a string value and then hits the output-token cap,
    it leaves an unterminated string and unclosed objects/arrays (the classic
    "Unterminated string" decode error). We walk the text tracking string/escape
    state to find what is still open, drop a dangling escape, close an open
    string, strip a trailing comma or dangling key colon, and append the missing
    closers. This lets the fields generated before the runaway one (e.g. all the
    markers) still be parsed instead of losing the whole extraction.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()

    repaired = text
    if in_string:
        if escaped:
            # Cut off right after a lone backslash; drop it so we don't emit a
            # dangling escape when we add the closing quote.
            repaired = repaired[:-1]
        repaired += '"'
    repaired = re.sub(r"[\s,]+$", "", repaired)
    if repaired.endswith(":"):
        # Truncated right after a key, e.g. `"ki67_percent":` -> give it a value.
        repaired += " null"
    repaired += "".join(reversed(stack))
    return repaired


def parse_json_response(text: str) -> dict | list:
    cleaned = strip_code_fences(text)
    # 1) Plain parse, then 2) pull the JSON out of any thinking/prose wrapper.
    for candidate in (cleaned, _extract_json_blob(cleaned)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # 3) Last resort: the model likely looped inside a value and was cut off by
    # the token cap, leaving truncated JSON. Repair and retry — the full text
    # first so trailing complete fields survive, then the extracted blob.
    for candidate in (cleaned, _extract_json_blob(cleaned)):
        try:
            return json.loads(_repair_truncated_json(candidate))
        except json.JSONDecodeError:
            continue
    # Nothing recovered; re-raise the original decode error for the caller to
    # record this report as failed.
    return json.loads(cleaned)


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
    # Numbers/bools are safe as-is (and must not be apostrophe-prefixed).
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    # Neutralize spreadsheet formula injection (CWE-1236): a cell starting with
    # = + - @ (or a leading tab/CR) can execute as a formula in Excel/Sheets, and
    # LLM-derived free text (diagnosis, names, extra__* markers) flows in here.
    # The raw values stay intact in the JSON/ZIP outputs.
    if text[:1] in ("=", "+", "-", "@", "\t", "\r"):
        text = "'" + text
    return text


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
    json_mode: str,
    schema: dict | None = None,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    """Send one prompt and return (raw model text, finish_reason).

    finish_reason is "length" when the model was cut off at the token cap
    (truncated output / likely a repetition loop), otherwise "stop". Callers use
    it to retry with a bigger budget or to flag the result as truncated.

    Both backends receive an identical (system, user) split so prompts and
    downstream JSON parsing stay uniform across providers.

    `json_mode` selects how an OpenAI-compatible server is asked to constrain
    its output:
      - "json_schema": grammar-constrain decoding to `schema` via
        `response_format={"type": "json_schema", ...}`. This is the only mode
        that actually forces the model to emit the schema's required fields, so
        it makes loosely-constrained runtimes (notably LM Studio's MLX engine)
        behave like the stricter ones (Ollama / llama.cpp GGUF). Without a
        `schema` it degrades to "json_object".
      - "json_object": only forces syntactically valid JSON (schema not
        enforced).
      - "off": no `response_format` at all.

    Servers that do not understand a given `response_format` answer with a
    400/422; we transparently step down to the next-weaker option so an older
    Ollama/llama.cpp build keeps working instead of hard-failing.
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
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            raise RuntimeError("The model declined to respond to this request (safety refusal).")
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        return text, ("length" if stop_reason == "max_tokens" else "stop")

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

    # Fallback ladder of response_format values, strongest first. We try each in
    # order and step down only when the server rejects the request itself
    # (400/422) — which is how OpenAI-compatible servers signal an unsupported
    # `response_format`. Other errors (auth, connection, 5xx) propagate as-is.
    response_formats: list[dict | None] = []
    if json_mode == "json_schema" and schema is not None:
        # Strip the `$schema` meta-key on a shallow copy: some structured-output
        # backends (incl. OpenAI strict mode) reject it, which would needlessly
        # drop us to json_object. The original `schema` is left untouched because
        # it is reused for validation and CSV building.
        schema_payload = {key: value for key, value in schema.items() if key != "$schema"}
        response_formats.append(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": schema_payload,
                    "strict": True,
                },
            }
        )
    if json_mode in ("json_schema", "json_object"):
        response_formats.append({"type": "json_object"})
    response_formats.append(None)

    last_error: Exception | None = None
    for response_format in response_formats:
        if response_format is None:
            request_kwargs.pop("response_format", None)
        else:
            request_kwargs["response_format"] = response_format
        try:
            completion = client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0]
            return (choice.message.content or ""), (choice.finish_reason or "stop")
        except (BadRequestError, UnprocessableEntityError) as exc:
            last_error = exc
    # Every response_format (including the unconstrained one) was rejected.
    if last_error is not None:
        raise last_error
    raise RuntimeError("Chat completion request failed.")


TRANSCRIBE_SYSTEM = (
    "You are a precise medical-document transcriber. You receive an image of a "
    "single page of a pathology / immunohistochemistry report. Transcribe ALL "
    "text exactly as printed, in reading order. Preserve the structure of marker "
    "result tables by keeping each marker on its own line next to its result "
    "(plain text, or a simple Markdown table). Do NOT interpret, summarize, "
    "translate, correct, or invent anything — output only what is on the page. "
    "Mark anything you cannot read as [illegible]."
)


def run_vision_transcription(
    *,
    backend: str,
    client,
    model: str,
    image_png: bytes,
    temperature: float,
    max_tokens: int,
) -> str:
    """Transcribe one rendered PDF page image to text via a vision-capable model.

    An alternative to OCR for scanned/image PDFs: the page is rendered to an image
    and a VLM reads it. The transcription then flows through the normal text
    pipeline (triage + schema-constrained extraction), so this is "smarter OCR"
    rather than image->JSON — which keeps the structured-output guarantees and
    reduces hallucination versus asking the VLM to fill the schema directly.
    """
    encoded = base64.b64encode(image_png).decode("ascii")
    user_text = "Transcribe this report page."
    if backend == "anthropic":
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=TRANSCRIBE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        },
                    ],
                }
            ],
        )
        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": TRANSCRIBE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
    )
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
    raw, _finish = run_chat_json(
        backend=backend,
        client=client,
        model=model,
        system=TRIAGE_SYSTEM,
        user=user,
        json_mode="off",
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


IHC_TERMS = ("immunohist", "imunohist", "inmunohist", "histoqu")
_CD_MARKER_RE = re.compile(r"\bcd\d+[a-z]*\b")
# Common non-CD immunohistochemistry markers, used as a baseline so the
# pre-screen still works when the schema has no dedicated marker fields
# (e.g. the free-form schema_fast.json).
BASE_IHC_MARKERS = frozenset({
    "alk", "bcl2", "bcl6", "beta f1", "ccr4", "ccr7", "cla", "cyclin d1",
    "eber", "ema", "foxp3", "gata3", "granzyme b", "granzyme m", "hla-dr",
    "icos", "ki-67", "ki67", "mib-1", "mum1", "oct2", "pax5", "perforin",
    "sox11", "sap", "tbx21", "tcl1", "tdt", "tia1", "lmp1", "myc",
})


def has_ihc_signal(text: str, marker_keywords: set[str]) -> bool:
    """True if the text shows a real immunohistochemistry signal.

    Matches an immunohistochemistry term (EN/PT/ES) or at least two distinct
    marker tokens. Marker matching is word-bounded so short marker names (EMA,
    ICOS, CLA, SAP, ...) do not match inside ordinary Portuguese/Spanish words
    such as "sistema", "edema" or "medicos". Any CDxx token counts as a marker
    even if it is not in the schema (e.g. CD20), so a report is recognised by
    its marker panel rather than a heading word.
    """
    low = text.lower()
    if any(term in low for term in IHC_TERMS):
        return True
    markers = set(_CD_MARKER_RE.findall(low))
    non_cd = BASE_IHC_MARKERS | {
        keyword for keyword in marker_keywords if not keyword.startswith("cd")
    }
    if non_cd:
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in non_cd) + r")\b")
        markers.update(pattern.findall(low))
    return len(markers) >= 2


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


VISION_RENDER_DPI = 200
VISION_MAX_PAGES = 10


def render_pdf_to_images(
    pdf_bytes: bytes, *, dpi: int = VISION_RENDER_DPI, max_pages: int = VISION_MAX_PAGES
) -> tuple[list[bytes], int]:
    """Render up to `max_pages` PDF pages to PNG bytes for a vision model or OCR.

    Reuses the already-imported PyMuPDF (fitz), so no new dependency is needed.
    Returns (images, total_page_count) so callers can warn when a long PDF was
    truncated to `max_pages`. `dpi` trades resolution (small marker-text
    legibility) against image token / OCR cost; 200 is a good default.
    """
    images: list[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = doc.page_count
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(min(total_pages, max_pages)):
            pixmap = doc.load_page(page_index).get_pixmap(matrix=matrix)
            images.append(pixmap.tobytes("png"))
        return images, total_pages
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


@st.cache_data(show_spinner=False)
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


TESSERACT_TO_APPLE_LANG = {
    "eng": "en-US",
    "ukr": "uk-UA",
    "rus": "ru-RU",
    "spa": "es-ES",
    "por": "pt-PT",
    "deu": "de-DE",
    "fra": "fr-FR",
    "ita": "it-IT",
}

APPLEVISION_MAX_PAGES = 50


def _apple_vision_languages(ocr_languages: str) -> list[str] | None:
    """Map the (tesseract-style) OCR-language field to Apple Vision BCP-47 codes."""
    codes = [c.strip() for c in (ocr_languages or "").replace("+", " ").split() if c.strip()]
    mapped = [TESSERACT_TO_APPLE_LANG.get(code, code) for code in codes]
    return mapped or None


def _apple_vision_page_text(png_bytes: bytes, languages: list[str] | None) -> str:
    from PIL import Image  # pulled in by ocrmac; imported lazily

    image = Image.open(io.BytesIO(png_bytes))
    # unit="line" returns one entry per text line (cleaner than the default
    # token granularity, which would scatter each word onto its own line).
    annotations = ocrmac_engine.OCR(
        image, language_preference=languages, unit="line"
    ).recognize()
    # Each annotation is (text, confidence, [x, y, w, h]) with NORMALIZED coords
    # and a BOTTOM-LEFT origin, so the top of the page has the LARGEST y. Sort by
    # descending y (top -> bottom), then ascending x, to recover reading order.
    ordered = sorted(annotations, key=lambda ann: (-round(ann[2][1], 2), ann[2][0]))
    return "\n".join(
        text for text, _conf, _box in ordered if text and text.strip()
    ).strip()


def run_apple_vision_ocr(
    *,
    pdf_bytes: bytes,
    ocr_languages: str,
    dpi: int = VISION_RENDER_DPI,
    max_pages: int = APPLEVISION_MAX_PAGES,
) -> tuple[str, int]:
    """OCR a PDF on-device with Apple's Vision framework (macOS).

    Returns (text, total_page_count). Renders pages with the shared fitz helper,
    runs Apple Vision per page, and joins with the "[Page N]" convention.
    """
    if not OCRMAC_AVAILABLE:
        raise RuntimeError(
            "Apple Vision OCR (ocrmac) is not installed. On macOS run: pip install ocrmac"
        )
    images, total_pages = render_pdf_to_images(pdf_bytes, dpi=dpi, max_pages=max_pages)
    languages = _apple_vision_languages(ocr_languages)
    parts: list[str] = []
    for page_number, png_bytes in enumerate(images, start=1):
        page_text = _apple_vision_page_text(png_bytes, languages)
        if page_text:
            parts.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(parts).strip(), total_pages


def prepare_pdf_report(
    *,
    report_name: str,
    pdf_bytes: bytes,
    pdf_input_mode: str,
    native_text_min_chars: int,
    ocr_languages: str,
    ocrmypdf_path: str | None,
    transcribe_images: Callable[[list[bytes]], str] | None = None,
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

    if pdf_input_mode in ("force_vision", "auto_vision_fallback"):
        run_vision = pdf_input_mode == "force_vision"
        if pdf_input_mode == "auto_vision_fallback":
            if native_chars < native_text_min_chars:
                run_vision = True
                warnings.append(
                    f"Native PDF text was short ({native_chars} chars), so a vision model was used."
                )
            if native_quality_reasons:
                run_vision = True
                warnings.append(
                    "Native PDF text looked low quality, so a vision model was used: "
                    + "; ".join(native_quality_reasons)
                    + "."
                )

        if not run_vision:
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=native_text,
                source_kind="pdf",
                text_extraction_method="pdf-native",
                preparation_warnings=warnings,
            )

        if transcribe_images is None:
            raise RuntimeError(
                "Vision mode was selected, but no vision model is configured for this run."
            )

        vision_text = ""
        try:
            page_images, total_pages = render_pdf_to_images(pdf_bytes)
            if total_pages > len(page_images):
                warnings.append(
                    f"PDF has {total_pages} pages; only the first {len(page_images)} "
                    "were sent to the vision model."
                )
            if page_images:
                vision_text = transcribe_images(page_images).strip()
                warnings.append(
                    f"A vision model transcribed {len(page_images)} page image(s)."
                )
            else:
                warnings.append("No page images could be rendered from this PDF.")
        except Exception as exc:
            warnings.append(f"Vision transcription failed: {exc}")

        if vision_text:
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=vision_text,
                source_kind="pdf",
                text_extraction_method="pdf-vision",
                preparation_warnings=warnings,
            )

        if native_text.strip():
            warnings.append(
                "Vision produced no usable text, so native PDF text was used instead."
            )
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=native_text,
                source_kind="pdf",
                text_extraction_method="pdf-native",
                preparation_warnings=warnings,
            )

        raise RuntimeError("No text could be extracted from this PDF (vision mode).")

    if pdf_input_mode in ("force_applevision", "auto_applevision_fallback"):
        run_av = pdf_input_mode == "force_applevision"
        if pdf_input_mode == "auto_applevision_fallback":
            if native_chars < native_text_min_chars:
                run_av = True
                warnings.append(
                    f"Native PDF text was short ({native_chars} chars), so Apple Vision OCR was used."
                )
            if native_quality_reasons:
                run_av = True
                warnings.append(
                    "Native PDF text looked low quality, so Apple Vision OCR was used: "
                    + "; ".join(native_quality_reasons)
                    + "."
                )

        if not run_av:
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=native_text,
                source_kind="pdf",
                text_extraction_method="pdf-native",
                preparation_warnings=warnings,
            )

        av_text = ""
        try:
            av_text, total_pages = run_apple_vision_ocr(
                pdf_bytes=pdf_bytes, ocr_languages=ocr_languages
            )
            if total_pages > APPLEVISION_MAX_PAGES:
                warnings.append(
                    f"PDF has {total_pages} pages; only the first {APPLEVISION_MAX_PAGES} were OCR'd."
                )
            av_text = av_text.strip()
            if av_text:
                warnings.append("Apple Vision (on-device) OCR was used.")
        except Exception as exc:
            warnings.append(f"Apple Vision OCR failed: {exc}")

        if av_text:
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=av_text,
                source_kind="pdf",
                text_extraction_method="pdf-applevision",
                preparation_warnings=warnings,
            )

        if native_text.strip():
            warnings.append(
                "Apple Vision produced no usable text, so native PDF text was used instead."
            )
            return PreparedReport(
                report_name=report_name,
                source_file_name=report_name,
                report_text=native_text,
                source_kind="pdf",
                text_extraction_method="pdf-native",
                preparation_warnings=warnings,
            )

        raise RuntimeError("No text could be extracted from this PDF (Apple Vision).")

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
            warnings.append("OCR (Tesseract/OCRmyPDF) text was used for this report.")

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
    json_mode: str,
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
        content, _finish = run_chat_json(
            backend=backend,
            client=client,
            model=model,
            system=(
                "You split source documents into distinct pathology reports. "
                "Return JSON only."
            ),
            user=prompt,
            # The splitter has its own {"reports": [...]} shape, not the
            # extraction schema, so it never gets schema-constrained output;
            # with schema=None, "json_schema" degrades to "json_object".
            json_mode=json_mode,
            schema=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = content or "{}"
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


MAX_OUTPUT_TOKENS_CEILING = 32000


def _bump_temperature(temperature: float) -> float:
    """Nudge a (possibly greedy) temperature up to break a repetition loop."""
    return round(min((temperature if temperature > 0 else 0.0) + 0.2, 1.0), 2)


def _candidate_size(parsed: dict | list) -> int:
    """Field/element count of a parsed candidate, used to keep the fullest partial."""
    if isinstance(parsed, (dict, list)):
        return len(parsed)
    return 0


def _normalize_for_grounding(text: str) -> str:
    # Collapse non-alphanumeric runs to single spaces (do NOT glue tokens, or a
    # marker like "CD3" followed by "100%" would read as "cd3100").
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower())


def _unverified_markers(parsed: dict | list, report_text: str) -> list[str]:
    """CD-type `additional_markers` whose name is absent from the source text.

    CD antigens are language-invariant, so a real one appears verbatim in the
    text; a CD name that is NOT there was almost certainly invented — most often
    an antibody clone NUMBER mistaken for a CD number (clone 124 -> "CD124").
    Scoped to the `CD<number>` pattern to avoid false-flagging translated or
    descriptive names (e.g. "Cyclin D1", "c-Myc") that legitimately differ from
    the source text. The trailing guard stops "CD5" matching inside "CD56".
    """
    if not isinstance(parsed, dict):
        return []
    markers = parsed.get("additional_markers")
    if not isinstance(markers, list):
        return []
    text_norm = _normalize_for_grounding(report_text)
    unverified: list[str] = []
    for entry in markers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("marker")
        if not isinstance(name, str) or not name.strip():
            continue
        name_norm = _normalize_for_grounding(name).strip()
        if not re.fullmatch(r"cd\d+\w*", name_norm):
            continue  # only audit language-invariant CD markers
        if not re.search(r"(?<![a-z0-9])" + re.escape(name_norm) + r"(?![0-9])", text_norm):
            unverified.append(name.strip())
    return unverified


@st.cache_data(show_spinner=False)
def cached_extract_outcome(
    *,
    _client,
    backend: str,
    base_url: str,
    model: str,
    instructions: str,
    schema_json: str,
    report_text: str,
    source_file_name: str,
    report_name: str,
    json_mode: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> ExtractionOutcome:
    """Extract one report, with retries, cached so reruns skip repeats.

    The cache key is every argument except `_client` (leading underscore ->
    excluded by st.cache_data), so changing the endpoint/model/schema/
    instructions/settings or the report text re-runs; an unchanged report on a
    rerun returns instantly (batch resume). Raises on total failure so transient
    failures are NOT cached — only usable outcomes are.
    """
    schema = json.loads(schema_json)
    prompt = f"""
Extract structured information from this pathology report.

Return exactly one JSON object that follows this JSON Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Report/source filename:
{source_file_name}

Report segment label:
{report_name}

Pathology report text:
{report_text}
""".strip()

    attempt_temp = float(temperature)
    attempt_max_tokens = int(max_tokens)
    last_error: str | None = None
    best: tuple[str, dict | list, list[str], bool] | None = None

    for _attempt in range(max(max_retries, 0) + 1):
        try:
            content, finish_reason = run_chat_json(
                backend=backend,
                client=_client,
                model=model,
                system=instructions,
                user=prompt,
                json_mode=json_mode,
                schema=schema,
                temperature=attempt_temp,
                max_tokens=attempt_max_tokens,
            )
        except Exception as exc:
            last_error = f"model call failed: {exc}"
            attempt_temp = _bump_temperature(attempt_temp)
            continue

        if not content.strip():
            last_error = "the model returned an empty response"
            attempt_temp = _bump_temperature(attempt_temp)
            if finish_reason == "length":
                attempt_max_tokens = min(attempt_max_tokens * 2, MAX_OUTPUT_TOKENS_CEILING)
            continue

        try:
            parsed = parse_json_response(content)
        except Exception as exc:
            last_error = f"could not parse JSON from the response: {exc}"
            attempt_temp = _bump_temperature(attempt_temp)
            if finish_reason == "length":
                attempt_max_tokens = min(attempt_max_tokens * 2, MAX_OUTPUT_TOKENS_CEILING)
            continue

        validation_errors = validate_instance(parsed, schema)
        truncated = finish_reason == "length"
        if not truncated:
            best = (content, parsed, validation_errors, truncated)
            break
        # Parsed but cut off mid-output: keep the MOST COMPLETE partial seen so
        # far, then (if retries remain) retry with a bigger budget + temp nudge.
        if best is None or _candidate_size(parsed) >= _candidate_size(best[1]):
            best = (content, parsed, validation_errors, truncated)
        last_error = "output was truncated at the token cap"
        attempt_max_tokens = min(attempt_max_tokens * 2, MAX_OUTPUT_TOKENS_CEILING)
        attempt_temp = _bump_temperature(attempt_temp)

    if best is None:
        raise RuntimeError(last_error or "extraction failed")

    content, parsed, validation_errors, truncated = best
    if truncated:
        status_label = "truncated"
    elif validation_errors:
        status_label = "schema-warning"
    else:
        status_label = "valid"

    # Deterministic grounding guard: a free-form marker whose name is absent from
    # the source text was likely inferred/fabricated (e.g. a clone code turned into
    # a bogus CD number). Flag for review and never let it claim high confidence.
    ungrounded = _unverified_markers(parsed, report_text)
    if ungrounded:
        if status_label in ("valid", "schema-warning"):
            status_label = "needs-review"
        if isinstance(parsed, dict):
            parsed["extraction_confidence"] = "low"
            note = (
                "auto-flagged: marker name(s) not found in the source text "
                "(possibly inferred from a clone): " + ", ".join(ungrounded)
            )
            prior = parsed.get("extraction_confidence_reason")
            parsed["extraction_confidence_reason"] = (
                f"{prior} | {note}" if isinstance(prior, str) and prior.strip() else note
            )

    return ExtractionOutcome(
        success=True,
        status_label=status_label,
        raw_response=content,
        parsed_json=parsed,
        validation_errors=validation_errors,
        error_message=None,
    )


def extract_reports(
    *,
    backend: str,
    client,
    base_url: str,
    model: str,
    instructions: str,
    schema: dict,
    prepared_reports: list[PreparedReport],
    json_mode: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 0,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    # Order-preserving schema string used as part of the per-report cache key.
    schema_json = json.dumps(schema, ensure_ascii=False)

    for prepared_report in prepared_reports:
        try:
            outcome = cached_extract_outcome(
                _client=client,
                backend=backend,
                base_url=base_url,
                model=model,
                instructions=instructions,
                schema_json=schema_json,
                report_text=prepared_report.report_text,
                source_file_name=prepared_report.source_file_name,
                report_name=prepared_report.report_name,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            results.append(
                ExtractionResult(
                    report_name=prepared_report.report_name,
                    source_file_name=prepared_report.source_file_name,
                    success=outcome.success,
                    status_label=outcome.status_label,
                    raw_response=outcome.raw_response,
                    parsed_json=outcome.parsed_json,
                    validation_errors=outcome.validation_errors,
                    source_kind=prepared_report.source_kind,
                    text_extraction_method=prepared_report.text_extraction_method,
                    preparation_warnings=prepared_report.preparation_warnings,
                    prepared_text=prepared_report.report_text,
                    error_message=outcome.error_message,
                )
            )
        except Exception as exc:
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
                    error_message=str(exc),
                )
            )

    return results


def is_local_endpoint(backend: str, base_url: str) -> bool:
    """True only when requests stay on this machine (loopback OpenAI-compatible).

    Anthropic and the blank/default OpenAI base URL are always remote.
    """
    if backend != "openai":
        return False
    url = (base_url or "").strip()
    if not url:
        return False  # blank -> OpenAI cloud default
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def render_results(results: list[ExtractionResult], schema: dict) -> None:
    st.subheader("Results")

    succeeded = sum(1 for r in results if r.success)
    warnings_n = sum(
        1 for r in results if r.status_label in ("schema-warning", "truncated", "needs-review")
    )
    failed = sum(1 for r in results if not r.success)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Reports", len(results))
    metric_cols[1].metric("Succeeded", succeeded)  # Succeeded + Failed == Reports
    metric_cols[2].metric("Warnings", warnings_n)
    metric_cols[3].metric("Failed", failed)
    st.caption("Showing the most recent extraction run.")

    summary_rows = [
        {
            "report": result.report_name,
            "source_file": result.source_file_name,
            "source": result.source_kind,
            "text_mode": result.text_extraction_method,
            "status": result.status_label,
            "confidence": (
                result.parsed_json.get("extraction_confidence")
                if isinstance(result.parsed_json, dict)
                else None
            ),
            "validation_errors": len(result.validation_errors),
        }
        for result in results
    ]
    st.dataframe(summary_rows, use_container_width=True)

    successful_csv_results = [
        result for result in results if result.success and isinstance(result.parsed_json, dict)
    ]
    results_csv = build_results_csv(successful_csv_results, schema)
    bundle = build_results_zip(results, results_csv=results_csv)

    download_cols = st.columns(2)
    if successful_csv_results:
        download_cols[0].download_button(
            "Download results (.csv)",
            data=results_csv,
            file_name="results.csv",
            mime="text/csv",
            key="download_results_csv",
        )
    else:
        download_cols[0].warning("No CSV rows available.")
    download_cols[1].download_button(
        "Download results (.zip)",
        data=bundle,
        file_name="medical_report_information_extractor_results.zip",
        mime="application/zip",
        key="download_results_zip",
    )

    only_problems = st.checkbox(
        "Show only problems (failed / warnings / truncated / needs-review / low confidence)",
        value=False,
        key="results_only_problems",
    )

    for result in results:
        parsed_dict = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        is_problem = (
            (not result.success)
            or result.status_label in ("schema-warning", "truncated", "needs-review")
            or parsed_dict.get("extraction_confidence") == "low"
        )
        if only_problems and not is_problem:
            continue
        with st.expander(result.report_name, expanded=is_problem):
            st.write(f"Source filename: `{result.source_file_name}`")
            st.write(f"Source kind: `{result.source_kind}`")
            st.write(f"Text extraction method: `{result.text_extraction_method}`")
            if result.preparation_warnings:
                st.warning("\n".join(result.preparation_warnings))
            if result.status_label == "truncated":
                st.warning(
                    "The model hit the output-token cap (possible repetition loop), so "
                    "this result may be truncated. Try raising Max output tokens or Max "
                    "retries, or nudging temperature up."
                )
            if result.status_label == "needs-review" or parsed_dict.get("extraction_confidence") == "low":
                _reason = parsed_dict.get("extraction_confidence_reason")
                st.warning(
                    "⚠️ Low confidence / needs review — verify the marker names against the "
                    "source text before trusting them. " + (_reason or "")
                )
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


# ---------------------------------------------------------------------------
# Lift engine (Datalab) — optional, self-contained, on-device PDF/image -> JSON.
#
# A separate tool from the text pipeline above. Instead of document -> text ->
# LLM, it renders page images and runs the datalab-to/lift 9B vision model
# in-process (HuggingFace). It is NOT model-agnostic: it only runs its own
# model, not the local Ollama/LM Studio LLM the text pipeline can use. Selected
# via the sidebar radio; when active it renders its own UI and st.stop()s, so
# the text-pipeline code below is left completely untouched.
# ---------------------------------------------------------------------------
ENGINE_TEXT = "Text pipeline (LLM)"
ENGINE_LIFT = "Lift — on-device PDF/image → JSON (Datalab)"

# Schema keys lift's decoder ignores; used only for a soft "guarantee weakened"
# warning (they don't stop extraction — the schema still runs).
LIFT_UNSUPPORTED_SCHEMA_KEYS = ("enum", "anyOf", "oneOf", "$ref", "additionalProperties")


@st.cache_resource(show_spinner="Loading the lift model (first run downloads ~18 GB from Hugging Face)…")
def get_lift_hf_model(device: str):
    """Load the datalab-to/lift 9B vision model in-process, once per session.

    Cached on `device` so the weights load a single time and are reused across
    reruns/files. `lift.settings.settings` is a singleton that lift's
    `load_model()` reads at construction time, so the device is pinned on it here
    (None -> device_map="auto"; "mps" -> Apple GPU; "cpu" -> slow fallback).
    """
    lift_settings.TORCH_DEVICE = device or None
    lift_settings.MODEL_CHECKPOINT = "datalab-to/lift"
    return LiftInferenceManager(method="hf")


def build_lift_result(
    out,
    *,
    report_name: str,
    source_file_name: str,
    source_kind: str,
    schema: dict,
    device: str,
) -> ExtractionResult:
    """Map a lift BatchOutputItem onto the app's shared ExtractionResult.

    lift returns extraction=None when the model output did not parse as JSON (the
    HF path is prompt-guided, not grammar-constrained, so that can happen). We
    reuse the app's jsonschema validator for the same valid / schema-warning
    labelling used by the text pipeline. The text pipeline's _unverified_markers
    grounding guard is intentionally NOT applied: it compares marker names against
    report text, which lift never produces.
    """
    parsed = out.extraction if isinstance(out.extraction, dict) else None
    success = (not out.error) and parsed is not None
    validation_errors = validate_instance(parsed, schema) if success else []
    if not success:
        status_label = "failed"
    elif validation_errors:
        status_label = "schema-warning"
    else:
        status_label = "valid"
    return ExtractionResult(
        report_name=report_name,
        source_file_name=source_file_name,
        success=success,
        status_label=status_label,
        raw_response=out.raw or "",
        parsed_json=parsed,
        validation_errors=validation_errors,
        source_kind=source_kind,
        text_extraction_method=f"lift-hf:{device}",
        preparation_warnings=[],
        prepared_text="(lift reads page images directly; no text stage)",
        error_message=None if success else (out.raw or "lift returned an error / unparseable JSON"),
    )


def run_lift_engine() -> None:
    """Render and run the self-contained on-device Lift extraction tool."""
    if not LIFT_AVAILABLE:
        st.info(
            "The Lift engine needs the optional `lift` package. Install the on-device "
            'backend with: `pip install "lift-pdf[hf]"` (pulls PyTorch / Transformers and, '
            "on first run, downloads the ~18 GB `datalab-to/lift` model)."
        )
        return

    with st.sidebar:
        st.header("Lift engine (Datalab)")
        st.success("Runs on-device — no report data leaves this Mac.")
        device = st.selectbox(
            "Device",
            ["mps", "cpu"],
            index=0,
            help="mps = Apple GPU (fast). cpu = fallback if MPS/bfloat16 misbehaves (much slower).",
        )
        lift_max_output_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=32000,
            value=12384,
            step=256,
            help="Upper bound on generated tokens for the whole document.",
        )
        lift_page_range = st.text_input(
            "Page range (optional)",
            value="",
            help='Limit PDF pages, e.g. "0-5,7". Blank = all pages.',
        )
        st.caption(
            "First run downloads the ~18 GB `datalab-to/lift` model. If it is gated, accept the "
            "license at https://huggingface.co/datalab-to/lift and run `huggingface-cli login`."
        )

    config_col, report_col = st.columns([1, 1])

    with config_col:
        st.subheader("Configuration")
        # Schema presets from config/, with lift's OWN session keys so the two
        # engines never overwrite each other's edited schema text.
        schema_choices = {path.name: path for path in sorted(CONFIG_DIR.glob("*.json"))}
        schema_names = list(schema_choices.keys()) or ["schema_lift.json"]
        default_name = (
            "schema_lift.json" if "schema_lift.json" in schema_names else schema_names[0]
        )
        selected_schema_name = st.selectbox(
            "Schema preset",
            schema_names,
            index=schema_names.index(default_name),
            help=(
                "Lift works best with a flat schema (no enum / unions / additionalProperties). "
                "schema_lift.json is a ready-made example."
            ),
        )
        if st.session_state.get("_lift_schema_preset") != selected_schema_name:
            st.session_state["lift_schema_text"] = json.dumps(
                load_json(schema_choices[selected_schema_name]), indent=2, ensure_ascii=False
            )
            st.session_state["_lift_schema_preset"] = selected_schema_name

        schema_file = st.file_uploader(
            "JSON Schema file (.json) — overrides the preset",
            type=["json"],
            key="lift_schema_upload",
        )
        if schema_file is not None:
            upload_id = f"{schema_file.name}:{schema_file.size}"
            if st.session_state.get("_lift_schema_upload") != upload_id:
                st.session_state["lift_schema_text"] = schema_file.getvalue().decode(
                    "utf-8", errors="replace"
                )
                st.session_state["_lift_schema_upload"] = upload_id

        schema_text = st.text_area("JSON Schema", key="lift_schema_text", height=320)

        lift_schema_obj = None
        try:
            lift_schema_obj = lift_resolve_schema(json.loads(schema_text))
        except json.JSONDecodeError as exc:
            st.error(f"Schema is not valid JSON: {exc}")
        except Exception as exc:
            st.error(f"Lift can't use this schema: {exc}")
        if any(key in (schema_text or "") for key in LIFT_UNSUPPORTED_SCHEMA_KEYS):
            st.warning(
                "This schema uses enum / anyOf / oneOf / $ref / additionalProperties, which lift's "
                "decoder ignores — the per-field guarantee is weakened. Prefer a flat schema like "
                "schema_lift.json."
            )

    with report_col:
        st.subheader("Reports")
        dep_cols = st.columns(2)
        dep_cols[0].metric("lift", "Found" if LIFT_AVAILABLE else "Missing")
        dep_cols[1].metric("Device", device)
        uploaded_pdfs = st.file_uploader(
            "Upload PDF report files (.pdf)",
            type=["pdf"],
            accept_multiple_files=True,
            key="lift_pdfs",
        )
        uploaded_images = st.file_uploader(
            "Upload image report files (.png/.jpg/.jpeg)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="lift_images",
        )
        st.info(
            "Lift sends rendered page images straight to the datalab-to/lift vision model and "
            "returns JSON matching your schema in a single pass. It does not use your local LLM."
        )

    run_lift = st.button("Extract with lift (on-device)", type="primary")

    if run_lift:
        uploads = [(f, "pdf") for f in (uploaded_pdfs or [])] + [
            (f, "image") for f in (uploaded_images or [])
        ]
        if lift_schema_obj is None:
            st.error("Fix the JSON Schema above before extracting.")
            st.stop()
        if not uploads:
            st.error("Upload at least one PDF or image.")
            st.stop()

        try:
            model = get_lift_hf_model(device)
        except ImportError:
            st.error('Install the on-device backend: pip install "lift-pdf[hf]"')
            st.stop()
        except Exception as exc:
            st.error(
                f"Could not load the lift model: {exc}\n\n"
                "If the model is gated, accept its license at "
                "https://huggingface.co/datalab-to/lift and run `huggingface-cli login`. "
                "If this looks like an MPS/bfloat16 error, switch Device to cpu."
            )
            st.stop()

        progress = st.progress(0.0)
        status = st.empty()
        results: list[ExtractionResult] = []
        for index, (uploaded, source_kind) in enumerate(uploads, start=1):
            status.write(f"Extracting {uploaded.name} ({index}/{len(uploads)})")
            try:
                with tempfile.TemporaryDirectory(prefix="mrie_lift_") as temp_dir:
                    file_path = Path(temp_dir) / uploaded.name
                    file_path.write_bytes(uploaded.getvalue())
                    out = lift_extract(
                        str(file_path),
                        lift_schema_obj,
                        model=model,
                        page_range=(lift_page_range.strip() or None),
                        max_output_tokens=int(lift_max_output_tokens),
                    )
                results.append(
                    build_lift_result(
                        out,
                        report_name=uploaded.name,
                        source_file_name=uploaded.name,
                        source_kind=source_kind,
                        schema=lift_schema_obj,
                        device=device,
                    )
                )
            except Exception as exc:
                results.append(
                    ExtractionResult(
                        report_name=uploaded.name,
                        source_file_name=uploaded.name,
                        success=False,
                        status_label="failed",
                        raw_response="",
                        parsed_json=None,
                        validation_errors=[],
                        source_kind=source_kind,
                        text_extraction_method=f"lift-hf:{device}",
                        preparation_warnings=[],
                        prepared_text="",
                        error_message=str(exc),
                    )
                )
            progress.progress(index / len(uploads))
        progress.empty()
        status.empty()
        st.session_state["lift_last_run"] = {"results": results, "schema": lift_schema_obj}

    if st.session_state.get("lift_last_run"):
        last_run = st.session_state["lift_last_run"]
        render_results(last_run["results"], last_run["schema"])


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

# Extraction engine selector. Selecting Lift runs a self-contained on-device tool
# and st.stop()s, so the text-pipeline UI below never renders for it.
engine = st.sidebar.radio(
    "Extraction engine",
    [ENGINE_TEXT, ENGINE_LIFT],
    help=(
        "Text pipeline: OCR / vision → text → your chosen LLM (OpenAI, Claude, local "
        "Ollama/LM Studio) with a JSON Schema. Lift: rendered page images straight into "
        "the datalab-to/lift 9B vision model, on-device (its own model, not your local LLM)."
    ),
)
if engine == ENGINE_LIFT:
    run_lift_engine()
    st.stop()

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

    remote_endpoint = not is_local_endpoint(backend, base_url)
    allow_phi_egress = True
    if remote_endpoint:
        # Bind consent to this specific endpoint: if the provider/base URL changes,
        # reset consent so PHI can't reach a new endpoint on a stale checkbox.
        endpoint_id = f"{provider_name}|{base_url.strip()}"
        if st.session_state.get("_phi_consent_endpoint") != endpoint_id:
            st.session_state["_phi_consent_endpoint"] = endpoint_id
            st.session_state["allow_phi_egress"] = False
        st.warning(
            "⚠️ This endpoint is **remote** — report text (and, in vision modes, page "
            "images) containing identifiable PHI will be sent off-device. This app does "
            "no de-identification."
        )
        allow_phi_egress = st.checkbox(
            "I understand and consent to sending PHI to this remote endpoint",
            key="allow_phi_egress",
        )

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
        json_mode = "off"
        temperature = 0.0
        st.caption("Temperature and JSON mode do not apply to the Anthropic backend.")
    else:
        json_mode = st.selectbox(
            "JSON output mode",
            ["json_schema", "json_object", "off"],
            index=0,
            format_func=lambda mode: {
                "json_schema": "Schema-constrained (forces every required field)",
                "json_object": "JSON object (valid JSON only, schema not enforced)",
                "off": "Off (no constraint)",
            }[mode],
            help=(
                "Schema-constrained sends your JSON Schema as "
                "`response_format=json_schema`, so the server grammar-constrains the "
                "model to emit every required field. This is what makes a local MLX "
                "model (e.g. in LM Studio) match the GGUF/Ollama result instead of "
                "dropping fields. It falls back to `json_object` automatically if the "
                "server does not support it. Use `json_object` or `off` for older servers."
            ),
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help=(
                "0.0 is greedy decoding — most reproducible, but under schema-constrained "
                "output some local models fall into a repetition loop and run to the token "
                "cap. If a model loops, raise this to 0.1–0.2 to break the loop with minimal "
                "loss of determinism."
            ),
        )
        anthropic_max_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=32000,
            value=8000,
            step=256,
            help=(
                "Hard cap on generated tokens so a runaway/repetition loop fails fast "
                "instead of filling the context window. Truncated output is repaired "
                "best-effort, so fields generated before the cut are still recovered. "
                "Raise it for very long reports."
            ),
        )

    max_retries = st.number_input(
        "Max retries on failure / truncation",
        min_value=0,
        max_value=3,
        value=1,
        step=1,
        help=(
            "If extraction fails, returns empty, or is cut off at the token cap, retry "
            "this many times with a nudged temperature and a doubled token budget. Helps "
            "with loop-prone local models."
        ),
    )

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
    pdf_mode_labels = {
        "auto_ocr_fallback": "Auto: native text, OCR if short or low quality",
        "native": "Native text only",
        "force_ocr": "Force OCR on all PDFs",
        "auto_vision_fallback": "Auto: native text, vision model if short or low quality",
        "force_vision": "Force vision model on all PDFs",
    }
    pdf_mode_options = [
        "auto_ocr_fallback",
        "native",
        "force_ocr",
        "auto_vision_fallback",
        "force_vision",
    ]
    if OCRMAC_AVAILABLE:
        pdf_mode_labels["auto_applevision_fallback"] = (
            "Auto: native text, Apple Vision OCR if short or low quality"
        )
        pdf_mode_labels["force_applevision"] = "Force Apple Vision OCR (on-device, macOS)"
        # On-device Apple Vision sits right after the Tesseract OCR options.
        pdf_mode_options[3:3] = ["auto_applevision_fallback", "force_applevision"]
    pdf_input_mode = st.selectbox(
        "PDF text preparation",
        pdf_mode_options,
        index=0,
        format_func=lambda value: pdf_mode_labels[value],
        help=(
            "OCR options use Tesseract/OCRmyPDF. Apple Vision (if available) is on-device "
            "macOS OCR. The vision options render each page to an image and ask a vision "
            "model (set below) to transcribe it. Either way the resulting text runs "
            "through the normal schema-constrained extraction."
        ),
    )
    ocr_languages = st.text_input(
        "OCR language(s)",
        value="eng",
        help=(
            "Tesseract codes joined with `+` (e.g. `eng+spa+por`). For Apple Vision these "
            "map to BCP-47 (eng→en-US, ukr→uk-UA, spa→es-ES, por→pt-PT)."
        ),
    )
    native_text_min_chars = st.number_input(
        "Min native PDF chars before OCR / vision",
        min_value=0,
        max_value=10000,
        value=80,
        step=20,
        help="In an auto mode, PDFs below this native-text threshold or with poor native layout are sent through OCR or the vision model.",
    )
    vision_model = st.text_input(
        "Vision model (for the vision PDF modes)",
        value="",
        help=(
            "Model used by the vision PDF modes above. Leave blank to reuse the main "
            "model. Point it at a vision-capable model served by your endpoint (e.g. a "
            "Qwen2.5-VL / Llama-Vision model in LM Studio or Ollama, or a Claude / "
            "GPT-4o model). The model must accept image input."
        ),
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

    schema_choices = {path.name: path for path in sorted(CONFIG_DIR.glob("*.json"))}
    schema_labels = {
        "schema.json": "Full — all dedicated marker fields (schema.json)",
        "schema_fast.json": "Fast / free-form — metadata + all-markers catch-all (schema_fast.json)",
        "schema_ukr.json": "Ukrainian — full (schema_ukr.json)",
        "schema_lift.json": "Lift-friendly — flat metadata + markers list (schema_lift.json)",
    }
    schema_names = list(schema_choices.keys()) or ["schema.json"]
    default_schema_index = (
        schema_names.index("schema.json") if "schema.json" in schema_names else 0
    )
    selected_schema_name = st.selectbox(
        "Schema preset",
        schema_names,
        index=default_schema_index,
        format_func=lambda name: schema_labels.get(name, name),
        help="Pick a built-in schema from the config folder. Editing the text below or uploading a file overrides it.",
    )
    # Load the chosen preset into the editable area whenever the selection changes.
    if st.session_state.get("_schema_preset_loaded") != selected_schema_name:
        st.session_state["schema_text"] = json.dumps(
            load_json(schema_choices[selected_schema_name])
            if selected_schema_name in schema_choices
            else default_schema,
            indent=2,
            ensure_ascii=False,
        )
        st.session_state["_schema_preset_loaded"] = selected_schema_name

    schema_file = st.file_uploader(
        "JSON Schema file (.json) — overrides the preset", type=["json"]
    )
    if schema_file is not None:
        # Only load the uploaded file when it actually changes (mirroring the
        # preset path); otherwise it would overwrite the user's manual edits on
        # every rerun, since the uploader keeps returning the same object.
        upload_id = f"{schema_file.name}:{schema_file.size}"
        if st.session_state.get("_schema_upload_loaded") != upload_id:
            st.session_state["schema_text"] = schema_file.getvalue().decode(
                "utf-8", errors="replace"
            )
            st.session_state["_schema_upload_loaded"] = upload_id

    schema_text = st.text_area(
        "JSON Schema",
        key="schema_text",
        height=320,
    )

with report_col:
    st.subheader("Reports")
    pymupdf_available = True
    ocrmypdf_path = shutil.which("ocrmypdf")
    tesseract_path = shutil.which("tesseract")

    dep_cols = st.columns(4)
    dep_cols[0].metric("PyMuPDF", "Found" if pymupdf_available else "Missing")
    dep_cols[1].metric("ocrmypdf", "Found" if ocrmypdf_path else "Missing")
    dep_cols[2].metric("tesseract", "Found" if tesseract_path else "Missing")
    dep_cols[3].metric("Apple Vision", "Found" if OCRMAC_AVAILABLE else "Missing")

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

    if remote_endpoint and not allow_phi_egress:
        st.error(
            "This endpoint is remote and PHI egress isn't confirmed. Tick the consent "
            "checkbox in the sidebar, or switch to a local endpoint."
        )
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
    def transcribe_pdf_images(page_images: list[bytes]) -> str:
        # Transcribe page by page so per-call image tokens stay bounded and the
        # output keeps the "[Page N]" convention used elsewhere in the pipeline.
        transcribed: list[str] = []
        for page_number, page_image in enumerate(page_images, start=1):
            page_text = run_vision_transcription(
                backend=backend,
                client=llm_client,
                model=(vision_model.strip() or model),
                image_png=page_image,
                temperature=float(temperature),
                max_tokens=int(anthropic_max_tokens),
            ).strip()
            if page_text:
                transcribed.append(f"[Page {page_number}]\n{page_text}")
        return "\n\n".join(transcribed).strip()

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
                    transcribe_images=transcribe_pdf_images,
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
                    json_mode=json_mode,
                    temperature=float(temperature),
                    max_tokens=16000,
                )
            )
        else:
            expanded_reports.append(prepared_report)
        progress.progress(index / max(len(prepared_reports), 1))

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
            try:
                # Always run the cheap LLM triage. The old keyword backstop saved
                # a call but could silently drop real reports whose marker names it
                # didn't recognise; correctness wins over one tiny classification.
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
            # Passed screening: switch the status back to the extraction message
            # (it was overwritten by the "Screening ..." line above).
            status.write(
                f"Extracting {prepared_report.report_name} ({index}/{len(expanded_reports)})"
            )

        single_result = extract_reports(
            backend=backend,
            client=llm_client,
            base_url=base_url,
            model=model,
            instructions=instructions_text,
            schema=schema_obj,
            prepared_reports=[prepared_report],
            json_mode=json_mode,
            temperature=float(temperature),
            max_tokens=int(anthropic_max_tokens),
            max_retries=int(max_retries),
        )[0]
        results.append(single_result)
        progress.progress(index / max(len(expanded_reports), 1))

    progress.empty()
    status.empty()

    # Persist results so downloads/widgets (which rerun the script with
    # run_extraction=False) don't wipe the Results panel. Rendering happens below,
    # outside the run guard, from session_state.
    st.session_state["last_run"] = {"results": results, "schema": schema_obj}


if st.session_state.get("last_run"):
    _last_run = st.session_state["last_run"]
    render_results(_last_run["results"], _last_run["schema"])
