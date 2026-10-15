# On-Camera Briefing: Daniel Okafor, CRO, Appeal-Summary Assistant Pilot Gate

**Target length:** 7–8 minutes (about 1,000 spoken words). **Format:** talking head, with the register's *Heat Map* and *Risk Register* tabs shared on screen where noted.
**Order:** recommendation first, then risks by priority rank (R01 → R12), then the decisions I need. Not chronological.

---

### 0. Bottom line (≈45 s)

> Daniel, the bottom line first. I recommend a **conditional go, in shadow mode only**.
>
> The model does what it was trained to do on letters that look like its training data: clean, well-structured summaries, and it never tells the reviewer to approve or deny. But it fails in the three places that matter most for a health plan. It misses urgent appeals. It drops crisis statements. And it does both more often for members who write in limited English and for older members.
>
> Two of the twelve risks are **pilot blockers** until we put deterministic guardrails in front of the model. I'll take them in priority order.

*[Screen: Heat Map tab, inherent vs residual]*

### 1. R01: Missed expedited appeals (Critical, 25) (≈75 s)

> Top risk. When a member's health is in jeopardy, we have 72 hours, not 30 days. The model correctly flagged only **62%** of expedited appeals. On my eight hand-written urgent scenarios (chemo starting Monday, two days of insulin left, out of seizure medication) it caught **two**.
>
> The most important detail: when the letter says *"my doctor says waiting could seriously jeopardize my health"*, the statement that legally requires expedited review, the model missed it **five times out of six**, even though that exact sentence was in its training data. It learned to react to words like "urgent", not to what the words mean.
>
> **Mitigation:** a rules engine screens every raw letter for urgency and can only escalate, never downgrade. The model's urgency call becomes advisory, and a UR nurse confirms urgency on every appeal during the pilot. **Owner:** Director of Utilization Management. With that in place, residual risk drops to Medium, but it depends on the guardrail, not the model.

### 2. R02: Crisis statements dropped (Critical, 20) (≈50 s)

> Second. I wrote letters containing a suicidal statement, possible elder neglect, insulin rationing with fainting, and domestic violence. The model surfaced **none of the four**. Part of this is design: the summary format has no field for safety concerns, so the model compresses them away. A reviewer who trusts the summary would never know.
>
> **Mitigation:** add a mandatory safety-flags field, run an independent screen on the raw letter that routes to Care Management's crisis protocol, and make it policy that the summary never replaces reading the letter. **Owner:** Medical Director, Behavioral Health. This is a pilot blocker.

### 3. R03: Language and age disparity (High, 16) (≈60 s)

> Third, fairness. I built a synthetic population of 3,000 members and tested the model across language, age and plan type. For members writing in limited English, urgent appeals were caught **41%** of the time, versus **68%** for standard English. Members 65 and older: **53%** versus **70%**. Both fail the four-fifths rule.
>
> I'll be candid: at this sample size the confidence intervals are wide. But the mechanism is clear. *"I only have three days of medication left"* was caught ten times out of ten. *"I have medicine only for three days more"*, the same fact in limited English, was caught zero out of five. Given our Section 1557 language-access obligations, that is a compliance issue, not just a model-quality one.
>
> **Mitigation:** retrain with limited-English and Spanish urgency expressions, and set a release gate that no slice may fall below 90% of the best slice. Spanish letters go to bilingual reviewers. **Owner:** Health Equity Officer.

### 4. R05: The model memorizes training records (High, 15) (≈45 s)

> Fourth, privacy. I planted a fake member record with a fake SSN four times in the training data. Afterward, the model ranked that secret **first out of a thousand** look-alike candidates, the maximum possible memorization score. The untrained model ranked it near the middle, which is chance. It didn't recite it verbatim yet, but four copies were enough.
>
> Today there is no real PHI in training. The moment we fine-tune on real appeals, this becomes a HIPAA pathway. **Mitigation:** de-identify and deduplicate training data, and make this canary test a release gate. **Owner:** Privacy Officer.

### 5. R04 and the research claim: Our scores are flattering (High, 15) (≈60 s)

> Fifth, how far to trust the numbers, including mine. You asked me to red-team a frontier claim. I chose Microsoft's report that GPT-4 beats the US medical licensing exam pass mark by more than 20 points, the kind of number vendors put in front of us. The improvement over earlier models is real. But one of the two exam sets is freely posted on the USMLE website, the paper's own memorization check is described by its authors as having *unknown recall*, and the text-only model scored 70 to 80 percent on image questions it couldn't see. It shows multiple-choice skill, not appeal-review reliability.
>
> I found the same problem in our own evaluation. Standard checks said our test set was clean: no duplicates, no shared members. But **99%** of test letters share long phrases with training letters. On differently written call notes, appeal-type accuracy fell from 97% to 51%, and evidence extraction from 97% to 48%. **Mitigation:** a contamination gate on every reported metric, and a held-out set of real de-identified appeals. **Owner:** Model Risk Management.

### 6. Remaining risks, briefly (≈45 s)

*[Screen: Risk Register, rows ranked 6–12]*

> The rest are Medium or below once mitigated:
>
> - **Fabrication (R06):** given an empty input, the model invented a whole appeal, including a member name and ID. Fix: input validation and a grounding check.
> - **Prompt injection (R07):** five of six attacks failed, but a hidden HTML comment downgraded an urgent appeal. Fix: sanitize inputs and run these probes on every release.
> - **Synthetic data (R10):** everything I've shown is synthetic, which is why I'm asking for shadow mode.
> - **Overreach, extraction gaps, vendor claims and our own tooling (R08, R09, R11, R12):** these are covered in the register. One note on R12: my first test run had a scoring bug that made the model look *less* safe than it is. I caught it by reading outputs. That's why harness testing is in the register too.

### 7. What I need from you (≈30 s)

> Three decisions:
>
> 1. Approve a **90-day shadow-mode pilot**: the model runs on de-identified real appeals, members are not affected, and results are compared against clinician summaries.
> 2. Confirm that **R01 and R02 are gating**, with the urgency rules engine and crisis screen live before shadow mode starts.
> 3. Name owners for R01, R02 and R03 by next week so the guardrails can be built before the gate.
>
> The register, the audit report and the reproducible code are all in the package. Thank you.

---

#### Delivery notes
- Keep slides minimal. The Heat Map tab is the only visual you need at the start; switch to the Risk Register for section 6.
- Say numbers once, slowly. Lead each risk with the consequence for a member, then the number.
- If time is tight, cut section 6 to one sentence. Never cut sections 0–3.
