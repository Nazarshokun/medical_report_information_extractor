"""Instructions manager — write, save, version, and delete the extraction
*instructions* (the system prompt) as named presets, mirroring the Schema builder.

Presets are plain-text files under config/instructions/. The extractor page lists
them in its "Instructions preset" dropdown, so a preset saved here is immediately
usable there. Auto-discovered as a page (pages/), so `streamlit run app.py` finds it.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Instructions manager", page_icon=":material/edit_note:", layout="wide")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
INSTR_DIR = CONFIG_DIR / "instructions"
INSTR_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_FILE = CONFIG_DIR / "instructions.txt"


def presets() -> dict:
    return {p.stem: p for p in sorted(INSTR_DIR.glob("*.txt"))}


# Open with the current default loaded, so the page never looks "empty".
st.session_state.setdefault(
    "im_text", DEFAULT_FILE.read_text(encoding="utf-8") if DEFAULT_FILE.exists() else ""
)

st.title(":material/edit_note: Instructions manager")
st.caption(
    "Write and save the extraction instructions (the system prompt) as named presets. "
    "They appear in the **Instructions preset** dropdown on the extractor page."
)

edit_col, side_col = st.columns([3, 2], gap="large")

with side_col:
    st.subheader("Start from")
    saved = presets()
    if saved:
        pick = st.selectbox("Open a saved preset", list(saved), key="im_open_pick")
        if st.button(":material/folder_open: Load into editor", use_container_width=True):
            st.session_state["im_text"] = saved[pick].read_text(encoding="utf-8")
            st.rerun()
    else:
        st.caption("No saved presets yet.")
    if DEFAULT_FILE.exists():
        if st.button(":material/description: Load current default", use_container_width=True):
            st.session_state["im_text"] = DEFAULT_FILE.read_text(encoding="utf-8")
            st.rerun()

with edit_col:
    st.subheader("Instructions")
    st.text_area(
        "Instructions text",
        key="im_text",
        height=380,
        label_visibility="collapsed",
        placeholder="You are a precise medical-data extractor. Extract only what is present…",
    )
    st.caption(f"{len(st.session_state.get('im_text', '')):,} characters")

    st.text_input("Preset name", key="im_name", placeholder="my_instructions")
    can_save = bool((st.session_state.get("im_text") or "").strip()) and bool(
        (st.session_state.get("im_name") or "").strip()
    )
    if st.button(":material/save: Save preset", type="primary", disabled=not can_save):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (st.session_state.get("im_name") or "").strip()).strip("_")
        if not safe:
            st.error("Enter a preset name first.")
        else:
            (INSTR_DIR / f"{safe}.txt").write_text(st.session_state["im_text"], encoding="utf-8")
            st.success(
                f"Saved **config/instructions/{safe}.txt** — pick it in the extractor's "
                "Instructions preset dropdown.",
                icon=":material/check_circle:",
            )

# --- Manage / delete saved presets -------------------------------------------
st.subheader("Manage saved presets")

just_deleted = st.session_state.pop("_im_deleted", None)
if just_deleted:
    st.toast(f"Deleted {just_deleted}", icon=":material/check_circle:")
delete_error = st.session_state.pop("_im_delete_error", None)
if delete_error:
    st.error(f"Could not delete: {delete_error}")


@st.dialog("Delete instructions preset?")
def _confirm_delete(name: str) -> None:
    st.write(f"Permanently delete **config/instructions/{name}.txt**? This can't be undone.")
    confirm, cancel = st.columns(2)
    if confirm.button("Delete", type="primary", icon=":material/delete:", use_container_width=True):
        try:
            (INSTR_DIR / f"{name}.txt").unlink()
            st.session_state["_im_deleted"] = f"{name}.txt"
        except OSError as exc:
            st.session_state["_im_delete_error"] = str(exc)
        st.rerun()
    if cancel.button("Cancel", use_container_width=True):
        st.rerun()


saved = presets()
if saved:
    row = st.columns([3, 1], vertical_alignment="bottom")
    target = row[0].selectbox("Saved preset", list(saved), key="im_delete_pick")
    if row[1].button("Delete…", icon=":material/delete:"):
        _confirm_delete(target)
else:
    st.caption("No saved instruction presets to delete yet.")
