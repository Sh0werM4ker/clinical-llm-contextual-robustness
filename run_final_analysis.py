"""Portable entry point; the original statistical implementation is unchanged."""
from pathlib import Path
import argparse
import importlib.util

def main():
    parser=argparse.ArgumentParser(description="Run the final case-cluster analysis on an author-supplied analysis CSV.")
    parser.add_argument("--dataset",type=Path,required=True,help="Final adjudicated analysis CSV; not included in this code-only attachment")
    parser.add_argument("--output-dir",type=Path,required=True,help="Directory for derived CSV and JSON outputs")
    args=parser.parse_args()
    if not args.dataset.is_file():
        parser.error("Dataset file does not exist")
    script=Path(__file__).resolve().parent/"final_analysis"/"cluster_inference_original.py"
    spec=importlib.util.spec_from_file_location("cluster_inference_original",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATASET=args.dataset.resolve()
    module.OUTPUT_DIR=args.output_dir.resolve()
    module.main()

if __name__=="__main__":
    main()
