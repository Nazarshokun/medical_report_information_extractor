"""Schema builder — a visual, block-based editor for the extraction JSON Schema.

Each field is a card ("block"); the blocks transform into a JSON Schema live. Saving
writes a preset to config/, which the extractor page's "Schema preset" dropdown globs,
so a schema built here immediately shows up there.

This is a second page via Streamlit's auto-discovery multipage: it lives in pages/, so
`streamlit run app.py` picks it up and adds it to the sidebar page nav. The extractor
(app.py) is left completely untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Schema builder", page_icon=":material/build:", layout="wide")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
LEAF_TYPES = [
    "string", "number", "integer", "boolean",
    "string[]", "number[]", "integer[]", "boolean[]",
]

# State: a list of block ids. Each field's values live in per-block keyed widgets
# (key only, no `value=`, to avoid the value/session-state conflict). A friendly
# starter field is seeded before its widgets render.
st.session_state.setdefault("sb_ids", [0])
st.session_state.setdefault("sb_next_id", 1)
st.session_state.setdefault("sb_name_0", "final_diagnosis")
st.session_state.setdefault("sb_desc_0", "Final pathologic diagnosis")
st.session_state.setdefault("sb_req_0", True)


def add_field() -> None:
    st.session_state.sb_ids.append(st.session_state.sb_next_id)
    st.session_state.sb_next_id += 1


def remove_field(field_id: int) -> None:
    if field_id in st.session_state.sb_ids:
        st.session_state.sb_ids.remove(field_id)


def fields_to_schema(fields: list[dict]) -> tuple[dict, list[str]]:
    """Pure transform: field blocks -> (JSON Schema, list of problems)."""
    properties: dict = {}
    required: list[str] = []
    problems: list[str] = []
    seen: set[str] = set()
    for field in fields:
        name = (field.get("name") or "").strip()
        if not name:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            problems.append(f"'{name}' isn't a valid field name (use letters, digits, underscore).")
            continue
        if name in seen:
            problems.append(f"Duplicate field '{name}'.")
            continue
        seen.add(name)

        ftype = field.get("type") or "string"
        if ftype.endswith("[]"):
            leaf: dict = {"type": "array", "items": {"type": ftype[:-2]}}
        else:
            leaf = {"type": ftype}

        description = (field.get("description") or "").strip()
        if description:
            leaf["description"] = description

        enum_raw = (field.get("enum") or "").strip()
        if enum_raw and not ftype.endswith("[]"):
            values: list = [v.strip() for v in enum_raw.split(",") if v.strip()]
            if ftype in ("integer", "number"):
                try:
                    values = [int(v) if ftype == "integer" else float(v) for v in values]
                except ValueError:
                    problems.append(f"'{name}': allowed values must all be {ftype}s.")
                    values = []
            if values:
                leaf["enum"] = values

        properties[name] = leaf
        if field.get("required"):
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, problems


def build_schema() -> tuple[dict, list[str]]:
    """Gather the block values from widget state and transform them."""
    fields = [
        {
            "name": st.session_state.get(f"sb_name_{fid}", ""),
            "type": st.session_state.get(f"sb_type_{fid}", "string"),
            "description": st.session_state.get(f"sb_desc_{fid}", ""),
            "required": st.session_state.get(f"sb_req_{fid}", False),
            "enum": st.session_state.get(f"sb_enum_{fid}", ""),
        }
        for fid in st.session_state.sb_ids
    ]
    return fields_to_schema(fields)


st.title(":material/build: Schema builder")
st.caption(
    "Add fields as blocks — they transform into a JSON Schema live. Save it, then pick "
    "it from the **Schema preset** dropdown on the extractor page."
)

builder_col, preview_col = st.columns([3, 2], gap="large")

with builder_col:
    st.subheader("Fields")
    for fid in st.session_state.sb_ids:
        with st.container(border=True):
            top = st.columns([4, 3, 1], vertical_alignment="bottom")
            top[0].text_input("Field name", key=f"sb_name_{fid}", placeholder="e.g. ki67_percent")
            top[1].selectbox("Type", LEAF_TYPES, key=f"sb_type_{fid}")
            top[2].button(
                ":material/delete:", key=f"sb_del_{fid}", help="Remove this field",
                on_click=remove_field, args=(fid,),
            )
            st.text_input(
                "Description (guides the model)", key=f"sb_desc_{fid}",
                placeholder="What this field means / how to fill it",
            )
            bottom = st.columns([1, 3], vertical_alignment="center")
            bottom[0].checkbox("Required", key=f"sb_req_{fid}")
            bottom[1].text_input(
                "Allowed values (optional → enum)", key=f"sb_enum_{fid}",
                placeholder="positive, negative, equivocal",
            )
    st.button(":material/add: Add field", on_click=add_field)

schema, problems = build_schema()

with preview_col:
    st.subheader("JSON Schema")
    st.json(schema)
    for problem in problems:
        st.warning(problem, icon=":material/warning:")

    st.text_input("Preset name", key="sb_preset_name", placeholder="my_schema")
    can_save = bool(schema["properties"]) and not problems
    if st.button(":material/save: Save preset", type="primary", disabled=not can_save):
        raw = (st.session_state.get("sb_preset_name") or "").strip()
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
        if not safe:
            st.error("Enter a preset name first.")
        else:
            (CONFIG_DIR / f"{safe}.json").write_text(
                json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            st.success(
                f"Saved **config/{safe}.json** — open the extractor page and choose it "
                "in the Schema preset dropdown.",
                icon=":material/check_circle:",
            )
    if not schema["properties"]:
        st.caption("Add at least one named field to enable saving.")


# --- Manage / delete saved schemas -------------------------------------------
# schema.json is the extractor's hard-loaded default, so it is protected here;
# every other config/*.json preset can be deleted (with a confirm step).
PROTECTED = {"schema.json"}

st.subheader("Manage saved schemas")

# Surface the outcome of a delete performed on the previous run.
just_deleted = st.session_state.pop("_sb_deleted", None)
if just_deleted:
    st.toast(f"Deleted config/{just_deleted}", icon=":material/check_circle:")
delete_error = st.session_state.pop("_sb_delete_error", None)
if delete_error:
    st.error(f"Could not delete schema: {delete_error}")


@st.dialog("Delete schema?")
def _confirm_delete(name: str) -> None:
    st.write(f"Permanently delete **config/{name}**? This can't be undone.")
    confirm, cancel = st.columns(2)
    if confirm.button("Delete", type="primary", icon=":material/delete:", use_container_width=True):
        try:
            (CONFIG_DIR / name).unlink()
            st.session_state["_sb_deleted"] = name
        except OSError as exc:
            st.session_state["_sb_delete_error"] = str(exc)
        st.rerun()
    if cancel.button("Cancel", use_container_width=True):
        st.rerun()


deletable = [p.name for p in sorted(CONFIG_DIR.glob("*.json")) if p.name not in PROTECTED]
if deletable:
    row = st.columns([3, 1], vertical_alignment="bottom")
    to_delete = row[0].selectbox("Saved schema", deletable, key="sb_delete_pick")
    if row[1].button("Delete…", icon=":material/delete:"):
        _confirm_delete(to_delete)
    st.caption("`schema.json` is the extractor's default and is protected from deletion.")
else:
    st.caption("No deletable schemas yet — `schema.json` is the protected default.")
