# Round 6 — LaMP-4 multi-user User-LoRA (cross-task OPPU recipe replication)

**Date:** 2026-07-07 (execution + writeup); plan pinned earlier, 2026-06-19
**Plan:** `experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md`
**Gate:** none (plan decision #9 — descriptive replication, no pre-registered PASS/FAIL)

---

## TL;DR

Cross-task replication of R5's LaMP-3 OPPU recipe on LaMP-4 (headline generation), K=100 users:

| Condition | R-1↑ | MPT | MGT | Latency (s) | n |
|---|---|---|---|---|---|
| C2: A1-lamp + BM25 | 0.235 | 359 | 13.1 | 0.60 | 100 |
| C3: + User-LoRA (OPPU recipe, stacked) | **0.242** | 359 | 13.1 | 0.44 | 100 |
| Δ (C3 − C2) | **+0.007** | 0 | 0.0 | n/a (see caveats) | |

Mean per-user ROUGE-1 lift is small and positive but **not statistically significant** (paired-t p=0.199, Wilcoxon p=0.299, 95% bootstrap CI [−0.0023, +0.0178] spans zero). Per the plan's own pre-specified disposition thresholds (§"What the raw numbers will inform"), +0.007 falls inside the **"≈0 (within ±0.01)"** bucket — i.e. OPPU's own weak-task LaMP-4 result (+0.003 R-1 lift over RAG) reproduces on our stack. This is the expected outcome given the plan's stated prior; nothing here overturns R5's LaMP-3 confirmation of Q4, it just doesn't extend it to LaMP-4.

---

## Hypothesis recap

**Q4 (CLAUDE.md).** Does an additional per-user LoRA on time-ordered user history meaningfully personalize beyond the Task-LoRA + BM25 alone? Confirmed on LaMP-3 by R5 (2026-06-19). R6 asks whether the same recipe + scaffolding replicates on a second task.

**R6 form (pinned 2026-06-19).** Same OPPU recipe, same stacking, same eval methodology as R5, applied to LaMP-4's K=100 pool. No formal gate — the plan explicitly designed this as a methodology check with an informative-either-way outcome (plan §"Why this round exists").

---

## Pre-registration audit

All scaffolding artifacts were committed before their corresponding submit; all production runs landed at `git_dirty=false`. Commit chain:

| Step | Commit | Subject |
|---|---|---|
| 2 | `b5f885f` | build top 100 lamp-4 users |
| 3 | `aa235d6` | pre-user dataset build |
| 4 | `4966445` | pre token-sizing run (sub) |
| 4/5 | `9c64ee9` | sizing scripts, gen scripts (round6_t3_sizing.py, round6_gen_configs.py, OPPU template) |
| 6 | `574258e` | lamp 4 per-user lora smoke test |
| 6 fix | `ca152a7` | lamp 4 training config (logging_steps 10→1, after the smoke revealed single-point loss logs for low-step users) |
| 7 | `67ae523` | train 100 lamp4 users |
| 7 retry | `2af77ae` | retry training (46 users, tyr1 GPU-slot oversubscription) |
| 8 | `d06f437` | lamp4 per user eval smoke test |
| 9 | `3fc5c8b` | lamp 4 eval (200-job sweep) |
| 9 retry | `3c4f377` | lamp 4 eval retry (18 users, tyr1 again + 4 users, Blackwell incompatibility) |
| 10/11 | `69f79ff` | aggregate results from lamp4 experiments (aggregator, paired_compare_per_user.py, Blackwell `require_gpus` guard retrofitted onto all 6 GPU subs, retry2 sub) |

Production run provenance:
- Step 2 (top-100 selection): 253 eligible users (`n_test≥1 AND seen_by_a1_lamp=0`), min profile_size=17, max=1100 — a much smaller and more skewed pool than R5's LaMP-3 (~1,800 eligible, profile_size 405–987).
- Step 3 (100 per-user datasets): all 100 metas `task=LaMP_4`, `framing=profile_entries_bm25`, `bm25_k=4`; leakage check clean on all 100 (`snapshot_record_id` in each user's train-id set, never dev/test). u00000011 (rank 1 in this pool) collided with a pre-existing R2-B artifact; verified byte-identical reproduction before overwrite.
- Step 4 (T3 sizing): global max token length 969 (well under the 8192 ceiling) → `max_seq_length=1024` pinned — far below LaMP-3's 7168, reflecting LaMP-4's much shorter BM25-retrieved context.
- Step 7 (100 trainings): all 100 adapters structurally verified (`r=8`, `target_modules=[q_proj,v_proj]`, correct `base_adapter_path`, non-NaN loss). First submission: 54/100 clean, 46/100 failed with `CUDA error: CUDA-capable device(s) is/are busy or unavailable` — all 46 landed on execute node `tyr1`, consistent with that node being handed more concurrent GPU-job slots than physical GPUs. Retry (identical configs) succeeded 46/46.
- Step 9 (200 evals): first submission 182/200 clean. 18 failed — same `tyr1` device-busy signature. Retry: 14/18 succeeded, 4/18 failed differently (`CUDA error: no kernel image is available for execution on the device`, host `fornjoter`) — the documented Blackwell (sm_120) incompatibility from CLAUDE.md's "GPU capability ceiling" note, hit for the first time on a LaMP-4 sub (the Llama subs already carried the guard; these hadn't). Retrofitted `require_gpus = Capability >= 8.0 && Capability < 10.0` onto all 6 LaMP-4 GPU subs and retried the 4 — clean. Final: 200/200, `n_filtered_records` matches expected `n_test` for every user, gold byte-match clean across all 248 shared record IDs between C2/C3.
- Step 10 (aggregation, ran locally): `results/LaMP_4_test_round6_C{2,3}.predictions.jsonl`, 248 lines each, zero id/gold mismatches.
- Step 11 (per-user-mean comparison): `results/paired_compare_c2_a1lamp_bm25_vs_c3_a1lamp_userlora_bm25_round6_LaMP_4_test.{json,pairs.jsonl}`. `n=100` (users, not records — see Setup), no `passes_registered_gate` field (no gate this round, by design).

---

## Setup recap (no new design decisions)

All design axes pinned in the plan's decision table (items 1–15). No re-litigation. Briefly:

- **Task / data.** LaMP-4 time-split test, K=100 users from `data/lamp_user_stats/LaMP_4_top100_users.json` (eligible filter: `n_test≥1 AND seen_by_a1_lamp=0`; 253-user pool; top-100 by profile_size, min=17, max=1100).
- **Task-LoRA.** `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000` (same canonical checkpoint as every other round).
- **User-LoRA recipe (OPPU, carried over verbatim from R5).** r=8, q+v only, lora_alpha=16, dropout=0.05, LR=1e-5, weight_decay=1e-2, AdamW, cosine + 3% warmup, 3 epochs, effective_bs=8 (per_device=2, grad_accum=4).
- **max_seq_length.** 1024 (T3-pinned; the one hyperparameter that differs from R5's 7168).
- **logging_steps.** 1, not R5's 10 — 22/100 users here have fewer than 10 total training steps (profile_size as low as 17), so R5's logging granularity would have collapsed first/final loss to a single point for a fifth of the pool. Caught during the Step-6 smoke.
- **Eval-time system slot.** BM25 k=4, matching train shape.
- **Stacking.** A1-lamp loaded via `base_adapter` plumbing (not merged) — same as R5.
- **No gate.** Descriptive replication only (plan decision #9).
- **Aggregation level.** Per-user mean ROUGE-1, not record-weighted — LaMP-4 users contribute 1–25 test records each (248 total), unlike LaMP-3's exactly-1-per-user pool, so record-level pairing would over-weight high-volume users and violate the paired-t independence assumption. `paired_compare_per_user.py` (new script this round) scores per record then averages within-user before pairing.

---

## Result tables

### Per-condition aggregate (n=100 users, 248 paired records)

| | C2 (A1-lamp + BM25) | C3 (+ User-LoRA stacked) |
|---|---|---|
| Mean per-user ROUGE-1 | 0.2349 | 0.2416 |
| MPT (mean prompt tokens) | 359.1 | 359.1 |
| MGT (mean generated tokens) | 13.1 | 13.1 |
| Latency (s/example) | 0.60 | 0.44 (not comparable — see caveats) |

### LaTeX table (as landed in the paper, `sections/experiments/2026-06-18-per-user-lora-lamp3.tex`, `tab:r6-lamp4-multi`)

```latex
\begin{table}[H]
  \centering
  \caption{\lampfour{} (K=100 users, $248$ test records, $1$--$25$/user): \tasklora{}+BM25 vs.\ \tasklora{}+BM25+\userlora{} (\up{}).}
  \label{tab:r6-lamp4-multi}
  \begin{tabular}{@{}l S S S S S@{}}
    \toprule
    Condition & {\rougeone{}\up} & {MPT} & {MGT} & {Latency (s)} & {$n$} \\
    \midrule
    \tasklora{} + BM25              & {0.235} & {359} & {13.1} & {0.60} & {100} \\
    \tasklora{} + BM25 + \userlora{} & {\best{0.242}} & {359} & {13.1} & {0.44} & {100} \\
    \midrule
    {$\Delta$} & {+0.007} & {0} & {0.0} & & \\
    \bottomrule
    \end{tabular}
\end{table}
```

**Reads:**
- **MPT identical between C2 and C3** — same BM25 profile retrieval regardless of which adapter stack is attached, same zero-inference-prompt-overhead story as R5.
- **MGT identical** — the User-LoRA doesn't change output length behavior.
- **Latency: NOT comparable this round.** Unlike R5 (where latency also came out identical, 0.618s both conditions), this round's C2 and C3 jobs landed on a heterogeneous mix of cluster GPU hosts *in different proportions* — one host (`tyr`) served only 6 C2 jobs but 41 C3 jobs. The raw 0.60s vs 0.44s gap reflects host assignment, not adapter-stacking cost. We omit the latency Δ in both the paper table and here for that reason.

### Descriptives (n=100 per-user means)

- Paired t-test: t=1.294, **p=0.1986**
- Wilcoxon signed-rank: **p=0.2990**
- 95% percentile bootstrap CI (10,000 resamples, seed=0): **[−0.0023, +0.0178]** — spans zero
- Wins/ties/losses (per-user mean R-1, C3 vs C2): **19 users favor C3, 12 favor C2, 69 tied**

None of these clear conventional significance. Sanity check requested by the plan (Step 11 acceptance): diffs are not degenerate — 31/100 users have a nonzero per-user mean difference, ruling out an adapter-load no-op bug.

### OPPU side-by-side

OPPU (arXiv:2402.04401), LaMP-4:

| | OPPU non-personalized | OPPU RAG | OPPU per-user PEFT | This round C2 (A1-lamp + BM25) | This round C3 (+ User-LoRA) |
|---|---|---|---|---|---|
| R-1 | 0.187 | 0.196 | 0.199 | 0.235 | 0.242 |
| Δ vs RAG | −0.009 | (ref) | **+0.003** | (ours' C2 baseline; not directly comparable) | **+0.007 vs C2** |

Caveats on the side-by-side:
- **Absolute R-1 not directly comparable** — our C2/C3 baselines are already well above OPPU's numbers because they stack on top of our Task-LoRA (A1-lamp), which OPPU has no equivalent of; OPPU's "RAG" condition is closer to a from-scratch BM25-prompted base model.
- **Effect magnitude is in the same small-and-noisy regime.** OPPU's own reported LaMP-4 lift (+0.003 R-1) is their weakest-reported task result and (per their paper) is itself not a strong effect. Our +0.007 is nominally larger but well within what 100-user sampling noise could produce, and our own single-user LaMP-4 history (R1–R4: point estimates +0.003 / −0.004 / +0.010 / −0.018 R-1, all sub-MDE) already flagged this as the weak task on our stack before this round ran.

---

## Honest caveats / known confounds

1. **This is the expected outcome, not a surprise.** The plan's own "honest priors" section flagged LaMP-4 as likely null-or-small going in, citing both OPPU's own weak reported lift and our four single-user LaMP-4 rounds (R1–R4) all landing sub-MDE. R6 ran anyway as a methodology check (does the R5 scaffolding/recipe generalize mechanically to a second task), which it does — the null result is informative, not a failure of execution.
2. **K=100 users, but only 248 total records (2.48/user average), heavily skewed** — one user (u00000011) contributes 25 test records, most contribute far fewer, several contribute just 1. The per-user-mean aggregation (Setup, above) protects the paired-t from being dominated by high-volume users, but it also means many per-user means are themselves single-record point estimates with no internal averaging — noisy inputs to an already underpowered n=100 test.
3. **Latency is uninterpretable this round** (see Result table caveat) — a genuine measurement gap, not glossed over. If a controlled latency comparison matters later, it would need same-host pinning or averaging across enough hosts that the mix is condition-independent by construction.
4. **Two distinct infra failure modes surfaced and got fixed along the way**, neither a pipeline correctness issue: (a) `tyr1` GPU-slot oversubscription under a 100-job burst (workaround: retry, no code change — an inherent risk of large simultaneous Condor GPU bursts on this cluster); (b) Blackwell (sm_120) incompatibility, now guarded against via `require_gpus` on all LaMP-4 GPU subs (a real gap CLAUDE.md had already flagged as likely to eventually bite un-hardened subs — it did, here).
5. **No recipe ablation.** Per the plan, the OPPU recipe carried over verbatim from R5 without re-tuning for LaMP-4's very different profile-size distribution (17–1100 vs LaMP-3's 405–987) or much shorter sequences (max_seq_length 1024 vs 7168). A small positive-but-noisy result doesn't distinguish "the recipe doesn't transfer" from "the recipe would work better tuned."

---

## Disposition

Per the plan's pre-specified thresholds (§"What the raw numbers will inform"): mean Δ R-1 = +0.007 falls in the **"≈0 (within ±0.01)"** bucket — *"OPPU's LaMP-4 weak-task result reproduces on our stack; no positive signal beyond noise; project decision = stop or pivot."*

R5's LaMP-3 confirmation of Q4 stands unchanged and unrelitigated. This round doesn't extend that confirmation to LaMP-4, but per the plan's own framing that was always the modal expected outcome for this specific task, not a project-level setback. R7 disposition is explicitly not pre-committed (plan: "designed post-hoc based on R6 raw numbers") — no follow-up axis is queued from this round; Phase 3 (on-device) remains the primary active track per CLAUDE.md.

---

## Artifacts

- Plan: `experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md`
- Top-100 user pool: `data/lamp_user_stats/LaMP_4_top100_users.json`
- T3 sizing: `data/lamp_user_stats/LaMP_4_round6_t3_sizing.json`
- Per-user training corpora: `data/lamp_user_train_LaMP_4_<fp>_bm25k4.{jsonl,meta.json}` (100 pairs)
- Per-user training configs: `train/config/user_lora_lamp4_oppu_<fp>.json` (100 files, gitignored, regeneratable via `data/lamp_user_stats/round6_gen_configs.py`)
- Per-user User-LoRA adapters: `train/checkpoints/user_lora_lamp4_<fp>_oppu_seed0/final/` (100 adapters)
- Per-user eval results: `results/LaMP_4_test_a1_lamp_1ep_seed0_checkpoint-1000_*_bm25k4_seed0_user<fp>.{json,predictions.jsonl}` (200 files, C2 + C3)
- Consolidated predictions (Step 10): `results/LaMP_4_test_round6_C{2,3}.predictions.jsonl`
- **Comparison output (Step 11):** `results/paired_compare_c2_a1lamp_bm25_vs_c3_a1lamp_userlora_bm25_round6_LaMP_4_test.{json,pairs.jsonl}`
- Paper section (updated with these results): `overleaf/6a2b1ada3ba0566171e752a2/sections/experiments/2026-06-18-per-user-lora-lamp3.tex` (`tab:r6-lamp4-multi`)

---

## References

- R6 plan: `experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md`
- R5 plan + writeup (predecessor, LaMP-3, Q4 confirmed): `experiments/2026-06-18-user-lora-round5-lamp3-plan.md`, `experiments/2026-06-18-user-lora-lamp3-round5-multi.md`
- R1–R4 LaMP-4 single-user writeups (same weak-task prior, single-user scale): `experiments/2026-06-15-user-lora-lamp4-u00000011-round1.md`, `..round2-B.md`, `..round3-alpha.md`, `..round4.md`
- Canonical Task-LoRA: `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/`
- OPPU paper: arXiv:2402.04401 — LaMP-4 numbers: non-pers R-1 0.187, RAG R-1 0.196, OPPU R-1 0.199 (lift over RAG = +0.003).
- Memory: [[project_user_lora_round6_lamp4_design]], [[project_user_lora_round5_lamp3_design]], [[project_direction_user_lora_pivot]], [[feedback_research_collaboration]], [[feedback_never_clobber_results]], [[feedback_user_owns_commits_submits]], [[feedback_writeup_defer_stats]]
