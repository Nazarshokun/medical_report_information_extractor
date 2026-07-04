"""Domain types that need a stable module identity for caching.

`ExtractionOutcome` is the return value of the `@st.cache_data`-cached
`cached_extract_outcome()` in app.py. Streamlit pickles cached return values, and a
class defined in the re-executed main script (module ``__main__`` — and worse under a
multipage app) fails to pickle reliably ("Cannot serialize ... __main__.ExtractionOutcome").
Defining it in this module — imported once, never re-executed — gives it a stable
``extraction_types.ExtractionOutcome`` identity, so the cache serializes cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass


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
