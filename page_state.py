"""Keep widget values when the user switches pages.

Streamlit deletes a widget's value at the end of any run that doesn't draw that
widget. In this multipage app that meant opening the Schema builder wiped the
extractor's provider, model, temperature, instructions and schema (and coming back
to the builder wiped its unsaved fields). Re-saving a widget's key at the start of
a run turns it into user state, which Streamlit keeps — so every page calls
`keep_widget_state()` right after `st.set_page_config`.

Only settable widgets belong here: buttons, download buttons and file uploaders
can't be written through `st.session_state` (and uploaded files can't be kept).
"""

from __future__ import annotations

import streamlit as st

# Extractor page (app.py): sidebar settings, the Configuration column, the pasted
# report and the results filter.
EXTRACTOR_KEYS = frozenset({
    "provider", "base_url", "api_key", "allow_phi_egress",
    "model_filter", "model_select", "model_text",
    "json_mode", "temperature", "max_tokens_openai", "max_tokens_anthropic",
    "max_retries", "split_multi_report_files", "prescreen_skip",
    "pdf_input_mode", "ocr_languages", "native_text_min_chars", "vision_model",
    "instructions_preset", "instructions_text", "schema_preset", "schema_text",
    "pasted_report", "save_ocr_reusable", "results_only_problems",
})
# Unsaved work on the Schema builder and Instructions manager pages.
PAGE_KEYS = frozenset({"sb_preset_name", "im_text", "im_name"})
# Schema-builder field blocks (sb_name_<id>, ...). Not sb_del_<id>: those are buttons.
PAGE_KEY_PREFIXES = ("sb_name_", "sb_type_", "sb_desc_", "sb_req_", "sb_enum_")


def keep_widget_state() -> None:
    """Re-save every persisted widget key so page switches don't delete it."""
    for key in list(st.session_state.keys()):
        if key in EXTRACTOR_KEYS or key in PAGE_KEYS or key.startswith(PAGE_KEY_PREFIXES):
            st.session_state[key] = st.session_state[key]
