# CST 627 Week 6: Red-Teaming a Frontier Claim and Auditing an Adapted Model

Meridian Health Plan scenario: governance review of an assistant that drafts member-appeal summaries for
utilization-review staff, ahead of the pilot gate with CRO Daniel Okafor.

## Deliverables

| Deliverable | File |
|---|---|
| Red-team review of a frontier research claim (Nori et al. 2023, GPT-4 on USMLE) | [reports/01_red_team_frontier_claim.md](reports/01_red_team_frontier_claim.md) |
| Model audit report (safety probes, fairness, contamination) | [reports/02_model_audit_report.md](reports/02_model_audit_report.md) |
| Risk register (NIST AI RMF + ISO/IEC 42001, mitigations, owners, residual risk) | [governance/Meridian_AI_Risk_Register.xlsx](governance/Meridian_AI_Risk_Register.xlsx) |
| On-camera briefing script, ordered top risk first | [reports/03_briefing_script.md](reports/03_briefing_script.md) |

## Layout

```
data/generate_population.py   synthetic member population + appeal letters (+ planted canary)
training/sft_train.py          LoRA SFT (adapted from the Week 3 pipeline)
audit/evaluate.py              task metrics, in-distribution and out-of-template
audit/safety_probes.py         43 self-authored safety probes with automated checks
audit/fairness.py              slice metrics, four-fifths flags, bootstrap CIs
audit/contamination.py         n-gram overlap, near-duplicates, member leakage, canary exposure
audit/rescore.py               re-score saved generations after a scoring-rule change
governance/build_risk_register.py   builds the Excel register from results/
results/                       audit outputs (generations, summaries, fairness, contamination)
```

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
bash run_audit.sh
```

All member data is synthetic. Tested on Windows 11 with an 8 GB RTX PRO 2000 Blackwell GPU. Training takes
about 15 minutes; the full audit takes about 30 minutes.
