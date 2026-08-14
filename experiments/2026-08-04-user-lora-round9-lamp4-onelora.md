# User-LoRA Round 9 — LaMP-4 re-run on One-LoRA FT

Executed 2026-08-04. Plan: `experiments/2026-07-20-user-lora-round9-lamp4-onelora-plan.md`
(pinned 2026-07-20, sat unbuilt for two weeks while R10-R14/PT3-PT7 and the
Warm-Start round ran ahead of it).

## Hypothesis

R9 closes the last gap in the R-track. Every LaMP task except LaMP-4 had already
been run with a per-user User-LoRA stacked on **One-LoRA FT** (the 7-task mixed
Task-LoRA): R8 covered LaMP-3, R10-R14 covered the five late tasks. LaMP-4 was
pinned and skipped.

The question is narrow, and the plan framed it carefully: R6 already found LaMP-4's
personalization effect to be a weak, non-significant null on the A1-lamp base
(ΔR-1 +0.007, p=0.20). So this round is **not** "does the lift survive" — there is
no lift to survive. It is "does the null hold in the same shape."

That matters because of what R8 found on LaMP-3: swapping A1-lamp → One-LoRA FT
collapsed a real, previously-confirmed lift (R5's ΔMAE −0.050) to *exactly* 0.000.
If the base swap is generally hostile to personalization, LaMP-4 should drift
negative here too. If R8's finding is specifically about degrading an *existing*
effect, LaMP-4's null should look roughly like R6's null.

## Setup

Everything except the base adapter is reused byte-identical from R6:

| Component | Source |
|---|---|
| User pool (K=100) | `data/lamp_user_stats/LaMP_4_top100_users.json` — R6's, unchanged |
| Per-user training data | `data/lamp_user_train_LaMP_4_<fp>_bm25k4.jsonl` — R6's, unchanged |
| Recipe | OPPU, byte-identical to R6/PT2: r=8, α16, dropout 0.05, q_proj+v_proj, AdamW LR=1e-5, wd=1e-2, 3 epochs, cosine + 3% warmup, batch 2 × grad-accum 4, `max_seq_length=1024`, `logging_steps=1` |
| Base adapter | **`train/checkpoints/a2_lamp_1ep_seed0/final` (One-LoRA FT)** — the only change |
| Eval | BM25 k=4, greedy, seed 0, `enable_thinking=False` |

`base_adapter_mode` is set explicitly to `"merge"` — `train.py`'s default, and the
behaviour every R/PT round already had implicitly. Stated in the template only
because the field exists as of 2026-08-01; R9 is a cold zero-init User-LoRA on a
merged base, not a warm-start continuation.

**Pool reuse was verified, not assumed.** `data/lamp_train_mixed7_bm25k4.meta.json`
records `per_task_reused["LaMP_4"] = true`: R7's `build_dataset_r7` read LaMP-4's
per-task training file read-only rather than rebuilding it, so the Task-LoRA saw
the identical LaMP-4 examples under A1-lamp and under One-LoRA FT, and R6's
leakage-eligibility computation still holds. This is the same argument R8 made for
LaMP-3, checked independently for LaMP-4 rather than inherited.

### Artifacts built

- `train/config/user_lora_lamp4_oppu_r9_template.json`
- `data/lamp_user_stats/round9_gen_configs.py` → 100 per-user configs (gitignored)
- `condor/gen_r9_subs.py` → all 5 subs (edit the generator, not the `.sub` files)
- `eval/aggregate_user_predictions_lamp4_r9.py`

Committed as `a2c019f` before any submit.

### Execution

| Stage | Cluster | Outcome |
|---|---|---|
| Train smoke (u00000070) | 179138 | pass |
| Train K=100 | 179139 | 99 exit-0 + 1 benign collision; **100/100 adapters validated** |
| Eval smoke (both arms) | 179142 | pass, 1/1 records matched |
| Eval 200 | 179169 | 198 exit-0 + 2 benign collisions |
| Aggregate | local | 248/248, gold byte-match passed |
| Paired compare | 179174 | pass |

**Zero genuine failures across 300 GPU jobs.** The `tyr1`/`modi` exclusions and the
Blackwell capability guard were baked into every sub from the first submit, per the
plan's row 11 — R8's 83/100 failure batch did not repeat. The 3 "failures" are all
refuse-to-overwrite collisions where the smoke run had already produced the correct
artifact for u00000070 (procs 99, and eval procs 99/199 — the last row of each block).

All 100 adapters were checked programmatically, not by exit code: r=8,
`target_modules=["q_proj","v_proj"]`, `base_adapter_path` ending in
`a2_lamp_1ep_seed0/final`, `base_adapter_mode="merge"`, and
`lora_weight_hash_before != lora_weight_hash_after` in every one.

## Result

**Mean ΔROUGE-1 +0.0026 — not significant.** Paired-t p=0.233, Wilcoxon p=0.076,
95% CI [−0.0015, +0.0069] spans zero. 19 wins / 72 ties / 9 losses over 100 users
(248 test records, per-user means).

All three LaMP-4 rounds, same pool, same 248 records, same comparison method:

| Round | Base adapter | Base R-1 | +User-LoRA | Δ | t_p | w_p | 95% CI | W/T/L |
|---|---|---|---|---|---|---|---|---|
| R6 | A1-lamp | 0.2349 | 0.2416 | **+0.0068** | 0.199 | 0.299 | [−0.0023, +0.0178] | 19/69/12 |
| PT2 | Per-Task-LoRA(LaMP-4) | 0.2575 | 0.2446 | **−0.0129** | 0.125 | 0.084 | [−0.0315, −0.0003] | 13/66/21 |
| R9 | One-LoRA FT | 0.2388 | 0.2414 | **+0.0026** | 0.233 | 0.076 | [−0.0015, +0.0069] | 19/72/9 |

Result files:
`results/paired_compare_c2_r9_bm25_vs_c3_r9_userlora_bm25_round6_LaMP_4_test.{json,pairs.jsonl}`,
consolidated predictions at `results/LaMP_4_test_r9_C{2,3}.predictions.jsonl`.

**The adapter did move the output: 56/248 predictions changed (22.6%).** That number
matters. R10-R14's headline was that the per-user adapter changed only 7.9% of
predictions overall, which made those nulls ambiguous — "personalization doesn't
help" and "the adapter is too weak to perturb anything" look identical from outside.
R9 is not in that regime. 22.6% sits squarely in the 14-28% band R10-R14 measured for
the *generation* tasks (vs 1-6% for the classification tasks). The adapter changed a
fifth of the headlines and the score did not move. This is a real null, not a
measurement floor.

## Deviations from the pinned plan

Two, both in the eval half, and both discovered **after** execution when the plan doc
was re-read. Recording them plainly rather than quietly:

1. **Eval job structure.** The plan (row 8) specified 101 jobs — C2″ as a single
   batch job via `--user-records-from-file`, C3″ as 100 per-user jobs. What ran was
   **200 jobs** (100 per-user C2 + 100 per-user C3), following PT2's structure.
2. **Comparison script.** The plan (row 9, scaffolding item 8) specified
   `eval/paired_compare.py` (flat, record-level). What ran was
   **`eval/paired_compare_per_user.py`** (grouped, per-user means) — R6's and PT2's
   script.

**The executed version is the correct one, and the plan's is wrong for this task.**
LaMP-4 users hold 1-25 test records each; 56 of the 100 hold more than one, 248 total.
The plan's rows 8-9 were written by analogy to R8/LaMP-3, where every user holds
exactly one test record and flat record-level comparison is identical to per-user
comparison. On LaMP-4 they are not identical: the flat route would have produced an
n=248 record-level statistic that is **not comparable to R6's headline +0.007**, which
is the very number the plan instructs to report side-by-side. The plan's own naming
chain gives it away — it predicts a 100-line consolidated C3 file, but 100 users hold
248 test records.

The deviation was not a considered call at build time; the scaffolding was cloned from
PT2 without reading the plan first, and happened to land on the right protocol because
PT2 is the same task. The verification that it *is* right (R6/PT2/R9 all n=100 over
248 records, confirmed from the result JSONs) was done afterwards. Two consequences
worth naming: the plan's stated artifact names (`LaMP_4_test_round9_C3.predictions.jsonl`,
`..._topK100.json`, `_r9_oppu_seed0` adapter dirs) do not match what is on disk, and
`condor/eval_lamp_user_lamp4_r9_baseline.sub` from the scaffolding inventory does not
exist because no separate baseline job was needed.

## Conclusion

**The null holds, and it holds in R6's shape, not PT2's.** R9's +0.0026 has the same
sign as R6's +0.0068, the same 19-win count, and a CI comfortably overlapping R6's.
Swapping A1-lamp → One-LoRA FT did not push LaMP-4 negative.

This is the plan's first branch: mild evidence that **R8's LaMP-3 collapse is about
degrading an existing real effect, not about universally suppressing personalization.**
One-LoRA FT is not uniformly hostile to per-user adapters — it specifically destroyed
LaMP-3's lift, and we still do not know why.

Set against the rest of the program, the base-adapter picture stays incoherent as a
general rule:

- **LaMP-3:** A1-lamp real lift → One-LoRA FT exactly zero (R8) → Per-Task partially
  preserved (PT1).
- **LaMP-4:** A1-lamp weak positive (R6) → One-LoRA FT weak positive (R9) → Per-Task
  flips negative (PT2).
- The other five tasks: null on both bases (R10-R14 / PT3-PT7).

LaMP-4 and LaMP-3 respond to the *same* base swap in opposite directions, and to the
*Per-Task* swap in opposite directions as well. The "task-mixing costs personalization"
hypothesis that R7/R8 pointed toward remains unsupported as a general rule. Q4 stays
LaMP-3-specific.

## Limitations

1. **Multiple comparisons.** R9 is the 22nd uncorrected comparison in this User-LoRA
   program. Nothing here reaches nominal significance so the point is moot for this
   round's own conclusion, but the Wilcoxon p=0.076 should not be read as "nearly
   significant" — at 22 tests it is unremarkable.
2. **A null cannot distinguish "no effect" from "effect below MDE."** R6 was already
   at/below its minimum detectable effect at K=100; R9 inherits that. The three LaMP-4
   Δ values (+0.0068, −0.0129, +0.0026) are all small relative to their CIs, and this
   round cannot rule out that they are three draws from one distribution centred near
   zero. Reading the R6→R9 agreement as *evidence of preservation* is only as strong
   as the CI overlap — it is consistency, not confirmation.
3. **Single seed** (0), as every round in this program.
4. **Provenance recorded `dirty=True`** on all 300 jobs, from an unrelated uncommitted
   modification to `condor/aggregate_warm.sub` left by a parallel session. The R9 code
   paths themselves were committed at `a2c019f` before any submit; the dirty flag does
   not indicate uncommitted R9 changes.
5. **PT2's CI excludes zero** ([−0.0315, −0.0003]) while its p-values do not reach
   0.05 — a bootstrap-CI/t-test disagreement already flagged in CLAUDE.md. It is not
   re-litigated here, but it is the one LaMP-4 number that should not be quoted as a
   clean null without that caveat.

## Follow-ups

- The R-track is now **complete across all 7 LaMP tasks**. No further base-swap rounds
  are queued.
- The open question R9 sharpens but does not answer: **why LaMP-3 specifically?** A
  targeted round would need to look at what One-LoRA FT does to LaMP-3's representation
  that it does not do to the other six, rather than adding more task × base cells.
- Paper: this round is not yet in `overleaf/.../sections/experiments/`. The natural home
  is `2026-07-31-user-lora-all-tasks.tex`, whose table already carries the twelve
  comparisons and would become thirteen with R9's row.
