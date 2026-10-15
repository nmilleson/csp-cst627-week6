"""Fairness audit across slices of the synthetic member population.

Reads results/<label>_<set>.jsonl (written by evaluate.py) and reports each
metric per slice, the worst/best ratio across groups (four-fifths rule flag
when < 0.8), and a bootstrap 95% CI on the gap between groups for two-group
slices. The headline metric is expedited recall: a missed expedited appeal
means a member waits up to 30 days instead of 72 hours.

Usage:
    python audit/fairness.py --labels base sft
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_jsonl  # noqa: E402

SLICES = ["preferred_language", "writing_style", "age_band", "plan_type", "filed_by"]
METRICS = ["expedited_recall", "urgency_match", "denial_reason_match", "service_match",
           "filed_by_match", "summary_f1", "valid_json"]


def slice_value(row, key):
    return row["gold"]["filed_by"] if key == "filed_by" else row["meta"][key]


def metric(rows, name):
    if name == "expedited_recall":
        pos = [r for r in rows if r["gold"]["urgency"] == "expedited"]
        return (sum(r["pred_urgency"] == "expedited" for r in pos) / len(pos)) if pos else None
    return sum(float(r[name]) for r in rows) / len(rows) if rows else None


def bootstrap_gap(a, b, name, n_boot=2000, seed=0):
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        ma = metric([rng.choice(a) for _ in a], name)
        mb = metric([rng.choice(b) for _ in b], name)
        if ma is not None and mb is not None:
            gaps.append(ma - mb)
    gaps.sort()
    return gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]


def audit(rows):
    out = {}
    for key in SLICES:
        groups = {}
        for r in rows:
            groups.setdefault(slice_value(r, key), []).append(r)
        entry = {"groups": {}}
        for g, grp in sorted(groups.items()):
            entry["groups"][g] = {"n": len(grp),
                                  "n_expedited": sum(r["gold"]["urgency"] == "expedited" for r in grp),
                                  **{m: metric(grp, m) for m in METRICS}}
        for m in METRICS:
            vals = {g: v[m] for g, v in entry["groups"].items() if v[m] is not None}
            if len(vals) < 2:
                continue
            lo_g, hi_g = min(vals, key=vals.get), max(vals, key=vals.get)
            ratio = vals[lo_g] / vals[hi_g] if vals[hi_g] else None
            info = {"worst_group": lo_g, "best_group": hi_g, "gap": vals[hi_g] - vals[lo_g],
                    "ratio": ratio, "four_fifths_flag": ratio is not None and ratio < 0.8}
            if len(groups) == 2:
                info["gap_ci95"] = bootstrap_gap(groups[hi_g], groups[lo_g], m)
                info["gap_significant"] = info["gap_ci95"][0] > 0
            entry.setdefault("disparity", {})[m] = info
        out[key] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=["base", "sft"])
    parser.add_argument("--sets", nargs="+", default=["eval_id", "eval_ood"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    report, flat = {}, []
    for label in args.labels:
        for name in args.sets:
            path = args.results_dir / f"{label}_{name}.jsonl"
            if not path.exists():
                continue
            res = audit(read_jsonl(path))
            report[f"{label}_{name}"] = res
            for key, entry in res.items():
                for g, vals in entry["groups"].items():
                    flat.append({"model": label, "eval_set": name, "slice": key, "group": g, **vals})

    with (args.results_dir / "fairness.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with (args.results_dir / "fairness_slices.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    for run, res in report.items():
        print(f"\n=== {run} ===")
        for key, entry in res.items():
            for m in ("expedited_recall", "urgency_match", "denial_reason_match", "summary_f1"):
                d = entry.get("disparity", {}).get(m)
                if not d:
                    continue
                groups = ", ".join(f"{g}={v[m]:.2f}" for g, v in entry["groups"].items() if v[m] is not None)
                flag = " <-- 4/5 FLAG" if d["four_fifths_flag"] else ""
                ci = f" CI95 gap {d['gap_ci95'][0]:+.2f}..{d['gap_ci95'][1]:+.2f}" if "gap_ci95" in d else ""
                print(f"{key:18s} {m:20s} {groups}{ci}{flag}")


if __name__ == "__main__":
    main()
