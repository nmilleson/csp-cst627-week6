"""Run a model over the in-distribution and out-of-template eval sets and score it.

Writes per-example rows (generation, scores, member slice attributes) to
results/<label>_<set>.jsonl and headline metrics to results/<label>_summary.json.
The fairness and contamination audits read these files.

Usage:
    python audit/evaluate.py --label base
    python audit/evaluate.py --label sft --adapter checkpoints/sft-adapter
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import generate, load_model, read_jsonl, score, write_jsonl  # noqa: E402

METRICS = ["valid_json", "has_all_fields", "urgency_match", "appeal_type_match", "filed_by_match",
           "denial_reason_match", "service_match", "evidence_f1", "argument_f1", "summary_f1",
           "decision_language"]


def expedited_recall(rows):
    pos = [r for r in rows if r["gold"]["urgency"] == "expedited"]
    return sum(r["pred_urgency"] == "expedited" for r in pos) / len(pos) if pos else None


def summarize(rows):
    n = len(rows)
    out = {"n": n, **{m: sum(float(r[m]) for r in rows) / n for m in METRICS}}
    out["expedited_recall"] = expedited_recall(rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--sets", nargs="+", default=["eval_id", "eval_ood"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(str(args.adapter) if args.adapter else None)
    summary = {"label": args.label}
    for name in args.sets:
        examples = read_jsonl(args.data_dir / f"{name}.jsonl")[: args.limit]
        start = time.time()
        generated = generate(model, tokenizer, examples[0]["instruction"], [ex["input"] for ex in examples])
        elapsed = time.time() - start
        rows = []
        for ex, gen in zip(examples, generated):
            gold = json.loads(ex["output"])
            rows.append({"meta": ex["meta"], "input": ex["input"], "gold": gold, "generated": gen,
                         **score(gen, gold)})
        write_jsonl(args.results_dir / f"{args.label}_{name}.jsonl", rows)
        summary[name] = {**summarize(rows), "seconds_per_example": elapsed / len(rows)}
        print(name, json.dumps(summary[name], indent=2))

    with (args.results_dir / f"{args.label}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
