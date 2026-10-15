# Red-Team Review: "GPT-4 exceeds the USMLE passing score by over 20 points"

**Prepared for:** Daniel Okafor, Chief Risk Officer, Meridian Health Plan
**Prepared by:** AI Governance Analyst
**Claim reviewed:** Nori, H., King, N., McKinney, S. M., Carignan, D., & Horvitz, E. (2023). *Capabilities of GPT-4 on Medical Challenge Problems.* arXiv:2303.13375 (v2, 12 Apr 2023). Microsoft / OpenAI.

## Why this claim matters to Meridian

Vendors pitching clinical and utilization-management AI routinely cite licensing-exam scores as evidence of "medical reasoning." If Meridian accepts a benchmark score as a proxy for appeal-review competence, it is relying on exactly the kind of claim reviewed here. The question for the pilot gate is: **what does a score like this license us to believe, and what does it not?**

## 1. The claim, stated precisely

> "GPT-4, without any specialized prompt crafting, exceeds the passing score on USMLE by over 20 points." (Abstract)

| Dataset | Source | Items | GPT-4 5-shot | GPT-4 0-shot | GPT-3.5 5-shot |
|---|---|---|---|---|---|
| USMLE Self Assessment | NBME paid practice forms (Step 1 Forms 25–30, Step 2 Forms 9–12, Step 3 Forms 1–5) | 2,173 | 86.65% | 83.76% | 53.61% |
| USMLE Sample Exam | Free PDFs on usmle.org (2021-10 versions) | 376 | 86.70% | 84.31% | 58.78% |

The "passing score" is taken from the USMLE website statement that examinees must answer "approximately 60 percent" of questions correctly (§3.1).

## 2. Measurement design

| Design element | What the paper did | Red-team note |
|---|---|---|
| Task format | Multiple-choice only; Step 3 computer-based case simulations (CCS) excluded (§6.3) | MCQ selection is not the same task as producing a clinical judgment or a free-text summary. |
| Modality | Text-only GPT-4; image/graph questions kept in but images were **not** shown to the model (§3.1.1) | 14.4% (SA) and 13.0% (Sample) of items reference media. |
| Prompting | Zero-shot and 5-shot with fixed templates; greedy decoding | Exemplar choice is reported to have little effect (Table 6). This is fine. |
| Scoring | Raw percent correct vs. "~60%" | Real USMLE pass/fail is a **scaled, equated** score. The "20 points" is percentage points against an approximate raw-score heuristic, not a pass/fail determination. |
| Uncertainty | No confidence intervals reported | With n = 2,173 the SA headline is statistically stable, but per-step results (Step 3 SA n = 176) carry roughly ±6 pt 95% CIs. |
| Contamination probe | MELD: feed the first half of each item, greedy-generate the second half, flag a Levenshtein ratio ≥ 0.95 (App. B) | Discussed in §4 below. |
| Model access | Released GPT-4 and a privileged GPT-4-base | Training data and cutoff are undisclosed, so independent replication of the contamination analysis is impossible. |

## 3. What the evidence genuinely supports

1. **Large relative improvement is real.** A 30+ point gain over GPT-3.5 on the same items with the same prompts is too large to be explained by noise or prompt choice. Even if contamination inflated both models, it is implausible that it explains the entire gap.
2. **High MCQ accuracy on NBME-style items.** On these specific items, GPT-4 selects the keyed answer about 84–87% of the time.
3. **Reasonable calibration on MCQ.** Confidence tracks accuracy better than GPT-3.5 (§5, Fig. 5.1). This is useful for any human-in-the-loop design.

The evidence does **not** support:

- that GPT-4 would *pass the USMLE* (scaled scoring, CCS excluded, approximate threshold);
- that GPT-4 can perform clinical or utilization-review reasoning in open-ended, real-world documents;
- that performance is robust to novel items unlike anything on the public internet.

## 4. Where contamination or leakage could explain the result

| # | Leakage pathway | Evidence in the paper | Assessment |
|---|---|---|---|
| C1 | **Sample Exam items are public.** They are free PDFs on usmle.org (URLs dated 2021-10), the same set Kung et al. used for ChatGPT, and widely reproduced on prep sites and forums. | The paper's argument that the material is "held behind an NBME paywall" (§6.2) applies only to the Self-Assessment forms, not the Sample Exam. | **High plausibility.** Public before the GPT-4 training cutoff. |
| C2 | **Paid NBME forms leak too.** Self-Assessment items and answer explanations are routinely transcribed into study decks, Reddit threads, and Q&A sites. | Not addressed. A paywall is not a provenance control. | **Moderate.** Paraphrased leakage is common in exam-prep ecosystems. |
| C3 | **MELD has "high precision but unknown recall"** (the authors' words). It detects only near-verbatim (≥95%) regeneration of the second half of an item. | The authors themselves "stress that this does not mean GPT-4 has not seen this data before." | **The negative result is weak evidence.** Memorizing the *question-to-answer mapping*, or seeing paraphrased items, would not trip MELD. |
| C4 | **Alignment can suppress verbatim regurgitation.** RLHF-tuned chat models are trained not to reproduce long verbatim text, so a black-box regurgitation test on the release model understates memorization. | The paper reports GPT-4-base scores 3–5% *higher* than the release model (Table 4). It does not say MELD was run on the base model. | **Moderate.** The probe was likely run on the configuration least likely to reveal memorization. |
| C5 | **Text-only model scores 70–80% on image questions it cannot see** (Table 3). | The authors attribute this to "logical reasoning and test-taking strategies" (Fig. 3.1). | **Ambiguous signal.** It is consistent with answer-pattern exploitation *or* with having seen the item plus its answer. Either way, it shows the score measures more than medical reasoning. |
| C6 | **Reliance on OpenAI's "contaminated vs. clean made no difference" finding** (§6.2, citing the GPT-4 Technical Report). | That analysis used a 50-character substring match, which also misses paraphrases. | **Low added assurance.** The same recall problem as MELD. |
| C7 | **MultiMedQA components (MedQA, MedMCQA, PubMedQA, MMLU)** are public GitHub/HF datasets since 2020–2021. | Contamination is not separately estimated for these. | **High plausibility** for the secondary benchmark claims. |

**Independent evidence that this matters.** Zhang et al. (2024, *GSM1k*, arXiv:2405.00332) rebuilt GSM8K with fresh, never-published items and found accuracy drops of up to 8 points for several model families (v4). Drop size was positively related to the model's probability of generating GSM8K examples (Spearman's r² = 0.36). This is direct evidence that public-benchmark scores can overstate capability, and that the size of the overstatement varies by model.

## 5. Contamination-robust re-test (what I would require before relying on such a claim)

1. **Fresh-item holdout.** Score on items written *after* the model's training cutoff (e.g., newly released NBME forms, or items commissioned from clinician item-writers), compared to the public items. A gap larger than about 5 points indicates contamination-driven inflation.
2. **Perturbation test.** Shuffle answer options, paraphrase stems, and change irrelevant numeric details. Memorized items degrade sharply; reasoned items do not.
3. **Answer-only probe.** Present the stem with the options removed and test whether the model produces the exact keyed answer string at above-chance rates.
4. **Base-model MELD and token-probability tests** (e.g., Min-K% probability) rather than release-model regeneration only.
5. **Report scaled-score equivalents with CIs**, and separate results for text-only and media items.

## 6. Bottom line for the pilot gate

The result **is evidence of strong multiple-choice performance on NBME-style items**. It **is not evidence** that a model can reliably summarize or reason about real member appeals. The paper's contamination analysis cannot rule out leakage, particularly for the public Sample Exam. **Meridian's policy should be that vendor benchmark scores are informative but never sufficient:** any model in the UR workflow must be validated on Meridian-specific, contamination-controlled data. That is what the audit of our own adapted model (Part 2) attempts. It found the same failure mode in miniature. Our in-distribution eval set is 99% "dirty" under the GPT-3 13-gram definition despite zero exact duplicates and zero member overlap. On out-of-template inputs, appeal-type accuracy falls from 0.97 to 0.51 and evidence-extraction F1 from 0.97 to 0.48.

## References

- Brown, T. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165 (13-gram contamination methodology, App. C).
- Carlini, N. et al. (2019). The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. USENIX Security. arXiv:1802.08232.
- Kung, T. H. et al. (2023). Performance of ChatGPT on USMLE. *PLOS Digital Health* 2(2): e0000198.
- Nori, H. et al. (2023). Capabilities of GPT-4 on Medical Challenge Problems. arXiv:2303.13375.
- OpenAI (2023). GPT-4 Technical Report. arXiv:2303.08774.
- Zhang, H. et al. (2024). A Careful Examination of Large Language Model Performance on Grade School Arithmetic. arXiv:2405.00332.
