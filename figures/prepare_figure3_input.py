"""Prepare plotting input from final analysis outputs; no new statistical analysis."""
from pathlib import Path
import argparse
import csv

def main():
    parser=argparse.ArgumentParser(description="Create the Figure 3 TSV from final analysis CSV outputs.")
    parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    strategies=[("evidence_first","Evidence-first"),("differential","Differential diagnosis"),("counter_anchor","Counter-anchor"),("self_check","Generic self-check")]
    definitions=[("A","part2_paired_cluster_inference_v3.csv","top1_correct"),("B","part2_paired_cluster_inference_v3.csv","evidence_override"),("C","failure_inclusive_cluster_inference_v3.csv","top1_correct")]
    output=[]
    for panel,name,outcome in definitions:
        with (args.results_dir/name).open(encoding="utf-8-sig",newline="") as handle:
            rows=list(csv.DictReader(handle))
        for strategy,label in strategies:
            matched=[r for r in rows if r["scope"]=="Part II" and r["comparison"]==strategy and r["outcome"]==outcome]
            if len(matched)!=1:
                raise ValueError(f"Expected exactly one result for {panel}/{strategy}")
            row=matched[0]
            output.append(dict(panel=panel,strategy=label,rd=100*float(row["paired_risk_difference"]),lo=100*float(row["case_cluster_bootstrap_low"]),hi=100*float(row["case_cluster_bootstrap_high"])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["panel","strategy","rd","lo","hi"],delimiter="\t")
        writer.writeheader();writer.writerows(output)

if __name__=="__main__":
    main()
