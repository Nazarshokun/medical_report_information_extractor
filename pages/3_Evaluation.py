"""Evaluation — a report card for the extractor against a gold-standard answer key.

Upload your hand-labelled *gold* JSONs and the extractor's *predicted* JSONs
(matched by filename), and this page compares them field-by-field with light
normalisation, reporting:
  - field-level accuracy (fraction of all fields correct),
  - full-document accuracy (fraction of reports where EVERY field is correct),
  - a per-field breakdown (which fields the model botches), and
  - every specific mismatch (report · field: expected vs got).

It's a pure scorer — no extraction happens here (that needs your configured provider
on the main page). Run extraction on the extractor page, download the results, then
drop the JSONs (or the whole results .zip) in here next to your gold answer key.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Evaluation", page_icon=":material/fact_check:", layout="wide")

_NUM_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*%?\s*$")


def _normalize(value):
    """Case/number/whitespace-insensitive form so 80 == '80%' and 'Positive' == 'positive'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if _NUM_RE.match(s):
            return float(re.search(r"-?\d+(?:\.\d+)?", s).group())
        return re.sub(r"\s+", " ", s)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    return value


def _match(gold_value, pred_value) -> bool:
    g, p = _normalize(gold_value), _normalize(pred_value)
    if isinstance(g, float) and isinstance(p, float):
        return abs(g - p) <= 1e-6
    if g in (None, "") and p in (None, ""):  # both "absent"
        return True
    return g == p


def _short(value, limit: int = 80) -> str:
    if value is None:
        return "(missing)"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _load_jsons(files, from_zip: bool = False):
    """Load {filename_stem: dict} from uploaded .json files (or .zip of them)."""
    loaded: dict = {}
    errors: list[str] = []
    for f in files or []:
        if from_zip and f.name.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(io.BytesIO(f.getvalue()))
                for member in archive.namelist():
                    if member.endswith(".json") and Path(member).name != "summary.json":
                        try:
                            loaded[Path(member).stem] = json.loads(archive.read(member).decode("utf-8"))
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"{member}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{f.name}: {exc}")
            continue
        try:
            loaded[Path(f.name).stem] = json.loads(f.getvalue().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{f.name}: {exc}")
    return loaded, errors


st.title(":material/fact_check: Evaluation")
st.caption(
    "Grade the extractor against a gold-standard answer key. Upload your labelled **gold** "
    "JSONs and the extractor's **predicted** JSONs (or its results `.zip`), matched by filename."
)

left, right = st.columns(2)
gold_files = left.file_uploader(
    "Gold JSONs (your answer key)", type=["json"], accept_multiple_files=True, key="ev_gold"
)
pred_files = right.file_uploader(
    "Predicted JSONs or results .zip (extractor output)",
    type=["json", "zip"], accept_multiple_files=True, key="ev_pred",
)

gold, gold_err = _load_jsons(gold_files)
pred, pred_err = _load_jsons(pred_files, from_zip=True)
for err in gold_err + pred_err:
    st.warning(f"Skipped an unreadable file — {err}", icon=":material/warning:")

common = sorted(set(gold) & set(pred))
only_gold = sorted(set(gold) - set(pred))

if not common:
    st.info(
        "Upload matching **gold** and **predicted** JSONs (same filename) to run the evaluation.",
        icon=":material/info:",
    )
    if gold or pred:
        st.caption(f"Loaded {len(gold)} gold and {len(pred)} predicted, but no filenames match.")
    st.stop()

if only_gold:
    shown = ", ".join(only_gold[:8]) + ("…" if len(only_gold) > 8 else "")
    st.caption(f"{len(only_gold)} gold report(s) had no matching prediction: {shown}")

# --- score ---
per_field: dict[str, list[int]] = {}   # field -> [correct, total]
mismatches: list[dict] = []
per_report: list[dict] = []
total_correct = total_fields = perfect_reports = 0

for stem in common:
    gold_doc, pred_doc = gold[stem], pred[stem]
    if not isinstance(gold_doc, dict):
        continue
    report_correct = 0
    fields = list(gold_doc.keys())
    for field in fields:
        gold_value = gold_doc.get(field)
        pred_value = pred_doc.get(field) if isinstance(pred_doc, dict) else None
        counters = per_field.setdefault(field, [0, 0])
        counters[1] += 1
        total_fields += 1
        if _match(gold_value, pred_value):
            counters[0] += 1
            report_correct += 1
            total_correct += 1
        else:
            mismatches.append(
                {"report": stem, "field": field,
                 "expected": _short(gold_value), "got": _short(pred_value)}
            )
    is_perfect = bool(fields) and report_correct == len(fields)
    perfect_reports += int(is_perfect)
    per_report.append(
        {"report": stem, "fields": len(fields), "correct": report_correct, "perfect": is_perfect}
    )

field_accuracy = total_correct / total_fields if total_fields else 0.0
doc_accuracy = perfect_reports / len(common) if common else 0.0

st.subheader("Scorecard")
cards = st.columns(4)
cards[0].metric("Field accuracy", f"{field_accuracy:.1%}", border=True)
cards[1].metric("Full-document accuracy", f"{doc_accuracy:.1%}", border=True)
cards[2].metric("Reports evaluated", len(common), border=True)
cards[3].metric("Fields evaluated", total_fields, border=True)

st.subheader("Per-field accuracy")
field_rows = sorted(
    (
        {"field": name, "accuracy": round(100 * correct / total, 1) if total else 0.0,
         "correct": correct, "total": total}
        for name, (correct, total) in per_field.items()
    ),
    key=lambda row: row["accuracy"],  # worst first, so problem fields surface at the top
)
st.dataframe(
    field_rows, hide_index=True, use_container_width=True,
    column_config={
        "field": st.column_config.TextColumn("Field", pinned=True),
        "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.0f%%"),
        "correct": st.column_config.NumberColumn("Correct"),
        "total": st.column_config.NumberColumn("Total"),
    },
)

st.subheader(f"Mismatches ({len(mismatches)})")
if mismatches:
    st.dataframe(
        mismatches, hide_index=True, use_container_width=True,
        column_config={
            "report": st.column_config.TextColumn("Report"),
            "field": st.column_config.TextColumn("Field"),
            "expected": st.column_config.TextColumn("Expected (gold)"),
            "got": st.column_config.TextColumn("Got (predicted)"),
        },
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["report", "field", "expected", "got"])
    writer.writeheader()
    writer.writerows(mismatches)
    st.download_button(
        "Download mismatches (.csv)", buffer.getvalue().encode("utf-8"),
        "eval_mismatches.csv", "text/csv",
    )
else:
    st.success("No mismatches — every graded field matched the gold answer key.", icon=":material/check_circle:")

with st.expander("Per-report detail"):
    st.dataframe(
        per_report, hide_index=True, use_container_width=True,
        column_config={
            "report": st.column_config.TextColumn("Report"),
            "fields": st.column_config.NumberColumn("Fields"),
            "correct": st.column_config.NumberColumn("Correct"),
            "perfect": st.column_config.CheckboxColumn("All correct"),
        },
    )

st.caption(
    "Matching is case-insensitive with number/whitespace normalisation (`80` = `80%`, "
    "`Positive` = `positive`, missing = null). Free-text fields are compared as normalised "
    "strings, so wording differences (e.g. `DLBCL` vs the full name) count as mismatches — "
    "keep your gold labels and the schema consistent. Only fields present in each gold file "
    "are graded; extra predicted fields are ignored."
)
