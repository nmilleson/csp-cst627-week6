"""Re-score saved generations after a change to the scoring rules in common.py.

Generation is the slow step; scoring is deterministic, so results/<label>_<set>.jsonl
can be re-scored in place and the summaries rebuilt without touching the GPU.

Usage:
    python audit/rescore.py --labels base sft
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_jsonl, score, write_jsonl  # noqa: E402
from evaluate import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=["base", "sft"])
    parser.add_argument("--sets", nargs="+", default=["eval_id", "eval_ood"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    for label in args.labels:
        summary_path = args.results_dir / f"{label}_summary.json"
        summary = json.load(summary_path.open(encoding="utf-8"))
        for name in args.sets:
            path = args.results_dir / f"{label}_{name}.jsonl"
            rows = [{**r, **score(r["generated"], r["gold"])} for r in read_jsonl(path)]
            write_jsonl(path, rows)
            summary[name] = {**summarize(rows), "seconds_per_example": summary[name]["seconds_per_example"]}
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(label, {n: round(summary[n]["decision_language"], 3) for n in args.sets})


if __name__ == "__main__":
    main()
