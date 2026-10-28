from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATING_DIR = PROJECT_ROOT / "ratings" / "pilot_advanced_local_clean_v1"
HUMAN_PATH = RATING_DIR / "final_analysis_dataset_final.csv"
PRIMARY_PATH = RATING_DIR / "primary_ai_judge_consolidated.xlsx"
SECONDARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "secondary_judge_glm47"
    / "pilot"
    / "secondary_judge_results.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "secondary_judge_glm47" / "pilot" / "comparison"

BINARY_FIELDS = [
    "top1_correct",
    "evidence_override",
    "mandatory_action_complete",
    "unsafe_management",
    "over_conservatism",
]
ORDINAL_FIELDS = ["gold_diagnosis_rank", "incorrect_anchor_rank", "anchor_adoption"]
CATEGORICAL_FIELDS = ["triage_level_scored"]
ALL_FIELDS = BINARY_FIELDS + ORDINAL_FIELDS + CATEGORICAL_FIELDS
TRIAGE_ORDER = ["routine", "urgent_outpatient", "same_day", "emergency"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - 64
    return index - 1


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                shared_strings.append("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_id = None
        for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{rel_ns}}}id")
                break
        if rel_id is None:
            raise RuntimeError(f"Sheet {sheet_name!r} not found in {path}")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships.findall(f"{{{pkg_rel_ns}}}Relationship"):
            if relationship.attrib.get("Id") == rel_id:
                target = relationship.attrib.get("Target")
                break
        if not target:
            raise RuntimeError(f"Worksheet relationship missing for {sheet_name!r}")
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = str(PurePosixPath("xl") / target)
        sheet_path = str(PurePosixPath(sheet_path))
        root = ET.fromstring(archive.read(sheet_path))

        rows: list[list[str]] = []
        for row in root.findall(f".//{{{main_ns}}}sheetData/{{{main_ns}}}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{{{main_ns}}}c"):
                index = column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{main_ns}}}t")
                    )
                else:
                    value_node = cell.find(f"{{{main_ns}}}v")
                    value = "" if value_node is None else (value_node.text or "")
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                    elif cell_type == "b":
                        value = "1" if value == "1" else "0"
                values[index] = value
            if values:
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])
    return rows


def rows_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    headers = rows[0]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        for row in rows[1:]
    ]


def load_primary() -> list[dict[str, str]]:
    results = rows_to_dicts(read_xlsx_sheet(PRIMARY_PATH, "Judge_Results_Raw"))
    linkage = rows_to_dicts(read_xlsx_sheet(PRIMARY_PATH, "Analysis_Linkage"))
    blinded_by_item = {row["judge_item_id"]: row["blinded_id"] for row in linkage}
    output = []
    for row in results:
        item = row["judge_item_id"]
        output.append({**row, "blinded_id": blinded_by_item[item]})
    return output


def normalize(value: str) -> str:
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def cohen_kappa(actual: list[str], predicted: list[str], labels: list[str] | None = None) -> float | None:
    if not actual:
        return None
    labels = labels or sorted(set(actual) | set(predicted))
    total = len(actual)
    observed = sum(a == p for a, p in zip(actual, predicted)) / total
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    expected = sum(actual_counts[label] * predicted_counts[label] for label in labels) / (total * total)
    if math.isclose(1 - expected, 0):
        return 1.0 if math.isclose(observed, 1) else None
    return (observed - expected) / (1 - expected)


def weighted_kappa(actual: list[str], predicted: list[str], labels: list[str]) -> float | None:
    if not actual:
        return None
    index = {label: position for position, label in enumerate(labels)}
    if len(labels) == 1:
        return 1.0
    total = len(actual)
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    observed_disagreement = sum(
        ((index[a] - index[p]) / (len(labels) - 1)) ** 2
        for a, p in zip(actual, predicted)
    ) / total
    expected_disagreement = 0.0
    for actual_label in labels:
        for predicted_label in labels:
            weight = ((index[actual_label] - index[predicted_label]) / (len(labels) - 1)) ** 2
            expected_disagreement += (
                weight
                * actual_counts[actual_label]
                * predicted_counts[predicted_label]
                / (total * total)
            )
    if math.isclose(expected_disagreement, 0):
        return 1.0 if math.isclose(observed_disagreement, 0) else None
    return 1 - observed_disagreement / expected_disagreement


def safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    tp = sum(a == "1" and p == "1" for a, p in zip(actual, predicted))
    tn = sum(a == "0" and p == "0" for a, p in zip(actual, predicted))
    fp = sum(a == "0" and p == "1" for a, p in zip(actual, predicted))
    fn = sum(a == "1" and p == "0" for a, p in zip(actual, predicted))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": recall,
        "specificity": safe_divide(tn, tn + fp),
        "precision": precision,
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "f1": f1,
    }


def compare(judge_name: str, judge_rows: list[dict[str, str]], human_by_id: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    judge_by_id = {row["blinded_id"]: row for row in judge_rows}
    common_ids = sorted(set(judge_by_id) & set(human_by_id))
    metrics: list[dict[str, Any]] = []
    disagreements: list[dict[str, str]] = []
    for field in ALL_FIELDS:
        pairs = [
            (identifier, normalize(human_by_id[identifier].get(field, "")), normalize(judge_by_id[identifier].get(field, "")))
            for identifier in common_ids
        ]
        pairs = [pair for pair in pairs if pair[1] != "" and pair[2] != ""]
        actual = [pair[1] for pair in pairs]
        predicted = [pair[2] for pair in pairs]
        labels = sorted(set(actual) | set(predicted))
        metric: dict[str, Any] = {
            "judge": judge_name,
            "field": field,
            "n": len(pairs),
            "exact_agreement": safe_divide(sum(a == p for a, p in zip(actual, predicted)), len(pairs)),
            "cohen_kappa": cohen_kappa(actual, predicted, labels),
            "quadratic_weighted_kappa": None,
        }
        if field in BINARY_FIELDS:
            metric.update(binary_metrics(actual, predicted))
        elif field in ORDINAL_FIELDS:
            order = [str(value) for value in range(0 if field == "anchor_adoption" else 1, 3 if field == "anchor_adoption" else 7)]
            metric["quadratic_weighted_kappa"] = weighted_kappa(actual, predicted, order)
        elif field == "triage_level_scored":
            metric["quadratic_weighted_kappa"] = weighted_kappa(actual, predicted, TRIAGE_ORDER)
        metrics.append(metric)
        for identifier, human_value, judge_value in pairs:
            if human_value != judge_value:
                disagreements.append(
                    {
                        "judge": judge_name,
                        "blinded_id": identifier,
                        "field": field,
                        "human_gold": human_value,
                        "ai_judge": judge_value,
                    }
                )
    return metrics, disagreements


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    human_rows = load_csv(HUMAN_PATH)
    human_by_id = {row["blinded_id"]: row for row in human_rows}
    primary_rows = load_primary()
    secondary_rows = load_csv(SECONDARY_PATH) if SECONDARY_PATH.exists() else []

    all_metrics: list[dict[str, Any]] = []
    all_disagreements: list[dict[str, str]] = []
    for name, rows in [("primary_web_gpt", primary_rows), ("secondary_local_glm47", secondary_rows)]:
        if not rows:
            continue
        metrics, disagreements = compare(name, rows, human_by_id)
        all_metrics.extend(metrics)
        all_disagreements.extend(disagreements)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "pilot_ai_vs_human_metrics.csv", all_metrics)
    write_csv(OUTPUT_DIR / "pilot_ai_vs_human_disagreements.csv", all_disagreements)
    summary = {
        "human_gold_records": len(human_rows),
        "primary_judge_records": len(primary_rows),
        "secondary_judge_records": len(secondary_rows),
        "metrics": all_metrics,
        "disagreement_count": len(all_disagreements),
        "note": "Human values are the final two-clinician ratings after third-clinician adjudication.",
    }
    (OUTPUT_DIR / "pilot_ai_vs_human_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
