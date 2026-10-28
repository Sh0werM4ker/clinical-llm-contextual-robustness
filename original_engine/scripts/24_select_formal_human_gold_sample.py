from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "ratings" / "formal_local_v1"
SEED = 20260721
SAMPLE_SIZE = 150


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    master = read_csv(RATINGS / "blinding_master.csv")
    rater_a = read_csv(RATINGS / "rater_A.csv")
    rater_b = read_csv(RATINGS / "rater_B.csv")
    if len(master) != 884 or len(rater_a) != 884 or len(rater_b) != 884:
        raise RuntimeError("Expected 884 rows in the formal master and both rater source files.")
    if rater_a != rater_b:
        raise RuntimeError("Rater A and B source files must be identical before scoring.")

    master_by_id = {row["blinded_id"]: row for row in master}
    rater_by_id = {row["blinded_id"]: row for row in rater_a}
    if set(master_by_id) != set(rater_by_id):
        raise RuntimeError("blinded_id mismatch between master and rater source files.")

    records: list[dict[str, str]] = []
    for row in master:
        combined = dict(row)
        combined["domain"] = rater_by_id[row["blinded_id"]]["domain"]
        records.append(combined)

    n = len(records)
    constraint_rows: list[np.ndarray] = []
    targets: list[float] = []
    labels: list[str] = []

    def add_constraint(label: str, predicate, target: int) -> None:
        vector = np.fromiter((1.0 if predicate(row) else 0.0 for row in records), dtype=float, count=n)
        if vector.sum() < target:
            raise RuntimeError(f"Insufficient candidates for {label}: available={int(vector.sum())}, target={target}")
        constraint_rows.append(vector)
        targets.append(float(target))
        labels.append(label)

    cases = sorted({row["case_id"] for row in records})
    models = sorted({row["logical_model_id"] for row in records})
    domains = sorted({row["domain"] for row in records})
    if len(cases) != 30 or len(models) != 3 or len(domains) != 6:
        raise RuntimeError("Expected 30 cases, 3 tested models, and 6 clinical domains.")

    add_constraint("total", lambda row: True, SAMPLE_SIZE)
    for case_id in cases:
        add_constraint(f"case:{case_id}:total", lambda row, c=case_id: row["case_id"] == c, 5)
        add_constraint(
            f"case:{case_id}:part1",
            lambda row, c=case_id: row["case_id"] == c and row["part"] == "part1",
            3,
        )
        add_constraint(
            f"case:{case_id}:part2",
            lambda row, c=case_id: row["case_id"] == c and row["part"] == "part2",
            2,
        )

    for model in models:
        add_constraint(f"model:{model}:total", lambda row, m=model: row["logical_model_id"] == m, 50)
        add_constraint(
            f"model:{model}:part1",
            lambda row, m=model: row["logical_model_id"] == m and row["part"] == "part1",
            30,
        )
        add_constraint(
            f"model:{model}:part2",
            lambda row, m=model: row["logical_model_id"] == m and row["part"] == "part2",
            20,
        )

    for domain in domains:
        add_constraint(f"domain:{domain}:total", lambda row, d=domain: row["domain"] == d, 25)
        add_constraint(
            f"domain:{domain}:part1",
            lambda row, d=domain: row["domain"] == d and row["part"] == "part1",
            15,
        )
        add_constraint(
            f"domain:{domain}:part2",
            lambda row, d=domain: row["domain"] == d and row["part"] == "part2",
            10,
        )

    for condition in ["no_label", "correct_label", "incorrect_label"]:
        add_constraint(
            f"part1_condition:{condition}",
            lambda row, c=condition: row["part"] == "part1" and row["anchor_condition"] == c,
            30,
        )
        for model in models:
            add_constraint(
                f"part1_condition_model:{condition}:{model}",
                lambda row, c=condition, m=model: (
                    row["part"] == "part1"
                    and row["anchor_condition"] == c
                    and row["logical_model_id"] == m
                ),
                10,
            )

    for intervention in ["evidence_first", "differential", "counter_anchor", "self_check"]:
        add_constraint(
            f"part2_intervention:{intervention}",
            lambda row, i=intervention: row["part"] == "part2" and row["intervention"] == i,
            15,
        )
        for model in models:
            add_constraint(
                f"part2_intervention_model:{intervention}:{model}",
                lambda row, i=intervention, m=model: (
                    row["part"] == "part2"
                    and row["intervention"] == i
                    and row["logical_model_id"] == m
                ),
                5,
            )

    add_constraint(
        "part1_repeat:1",
        lambda row: row["part"] == "part1" and row["repeat_id"] == "1",
        45,
    )
    add_constraint(
        "part1_repeat:2",
        lambda row: row["part"] == "part1" and row["repeat_id"] == "2",
        45,
    )

    matrix = csr_matrix(np.vstack(constraint_rows))
    target = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(SEED)
    objective = rng.random(n)
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, target, target),
        options={"time_limit": 120, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Human gold sample optimization failed: {result.message}")

    selected_indices = [index for index, value in enumerate(result.x) if value > 0.5]
    if len(selected_indices) != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} selected rows; found {len(selected_indices)}.")
    selected = [records[index] for index in selected_indices]
    rng.shuffle(selected)

    manifest_rows: list[dict[str, object]] = []
    selected_ids: list[str] = []
    for order, row in enumerate(selected, start=1):
        selected_ids.append(row["blinded_id"])
        manifest_rows.append(
            {
                "sample_order": order,
                "blinded_id": row["blinded_id"],
                "task_id": row["task_id"],
                "case_id": row["case_id"],
                "domain": row["domain"],
                "part": row["part"],
                "logical_model_id": row["logical_model_id"],
                "anchor_condition": row["anchor_condition"],
                "intervention": row["intervention"],
                "repeat_id": row["repeat_id"],
                "sampling_seed": SEED,
                "selection_method": "prespecified stratified MILP; independent of AI-judge labels",
            }
        )

    source_fields = list(rater_a[0].keys())
    selected_rater_rows = [rater_by_id[blinded_id] for blinded_id in selected_ids]
    write_csv(
        RATINGS / "human_gold_150_selection_manifest.csv",
        manifest_rows,
        list(manifest_rows[0].keys()),
    )
    write_csv(RATINGS / "human_gold_150_rater_A_source.csv", selected_rater_rows, source_fields)
    write_csv(RATINGS / "human_gold_150_rater_B_source.csv", selected_rater_rows, source_fields)

    qc = {
        "sampling_seed": SEED,
        "selection_method": "prespecified stratified mixed-integer linear optimization; no AI-judge label used",
        "sample_size": len(selected),
        "unique_blinded_ids": len(set(selected_ids)),
        "counts": {
            "case_id": dict(sorted(Counter(row["case_id"] for row in selected).items())),
            "domain": dict(sorted(Counter(row["domain"] for row in selected).items())),
            "part": dict(sorted(Counter(row["part"] for row in selected).items())),
            "model": dict(sorted(Counter(row["logical_model_id"] for row in selected).items())),
            "part1_anchor_condition": dict(sorted(Counter(
                row["anchor_condition"] for row in selected if row["part"] == "part1"
            ).items())),
            "part2_intervention": dict(sorted(Counter(
                row["intervention"] for row in selected if row["part"] == "part2"
            ).items())),
            "part1_repeat": dict(sorted(Counter(
                row["repeat_id"] for row in selected if row["part"] == "part1"
            ).items())),
        },
        "constraint_count": len(labels),
        "solver_message": result.message,
        "all_constraints_exact": bool(np.allclose(matrix @ (result.x > 0.5), target)),
    }
    (RATINGS / "human_gold_150_sampling_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
