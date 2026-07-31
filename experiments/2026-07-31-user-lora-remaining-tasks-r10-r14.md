# User-LoRA on the last 5 LaMP tasks — R10-R14 + PT3-PT7

**Status:** DONE 2026-07-31. All ten tracks executed end-to-end.
**Plan:** `experiments/2026-07-27-user-lora-remaining-tasks-plan.md` (pinned 2026-07-27 via `/grill_me`; scope extended to 5 tasks 2026-07-30).

## Hypothesis

Does per-user personalization (OPPU-recipe User-LoRA stacked on a Task-LoRA)
help on the five LaMP tasks that had never had a User-LoRA round — LaMP-2-news,
LaMP-1, LaMP-7, LaMP-2-movies, LaMP-5 — and does the answer depend on which
Task-LoRA it is stacked on (One-LoRA FT vs Per-Task-LoRA)?

This closes User-LoRA coverage on all 7 LaMP tasks. Descriptive reporting, no
pre-registered gate (convention since R6).

## Setup

Ten tracks: each task run on both bases in parallel. R-track base = One-LoRA FT
(`a2_lamp_1ep_seed0/final`), PT-track base = Per-Task-LoRA(task).

| track | task | K | training framing | max_seq_length | eval | metric |
|---|---|---|---|---|---|---|
| R10 / PT3 | LaMP-2-news | 27 | records + BM25 k=4 | 768 | grouped per-user | accuracy |
| R11 / PT4 | LaMP-1 | 100 | **unsupervised CLM** | 1792 | flat | accuracy |
| R12 / PT5 | LaMP-7 | 100 | **unsupervised CLM** | 256 | flat | rouge1 |
| R13 / PT6 | LaMP-2-movies | 100 | profile reframing + BM25 k=4 | 768 | flat | accuracy |
| R14 / PT7 | LaMP-5 | 100 | profile reframing + BM25 k=4 | 2048 | flat | rouge1 |

OPPU recipe unchanged and not relitigated (r=8, q_proj+v_proj, α=16,
dropout=0.05, LR=1e-5, wd=1e-2, 3 epochs, cosine + 3% warmup, per_device=2,
grad_accum=4). `max_seq_length` measured per task by its own T3 pass.

**854 User-LoRAs trained** (2×27 + 8×100), **1,354 paired test records** evaluated.

LaMP-1/LaMP-7 used the new `train/train_unsupervised_clm.py` (OPPU §3's
right-shifted-history objective: raw history text, no chat template, no role
mask, loss on every token). LaMP-1/LaMP-2-movies/LaMP-5 sourced their profile
snapshot from the user's own TEST record — for those three tasks **zero** users
hold both a train and a test record, so no other source exists. Eval is
unchanged in all cases (BM25 + the ordinary supervised prompt shape).

## Result

| track | task | base | metric | C2 | C3 | Δ | t_p | w_p | 95% CI | w/t/l |
|---|---|---|---|---|---|---|---|---|---|---|
| **R10** | LaMP-2-news | One-LoRA FT | acc | 0.8210 | 0.8308 | **+0.0098** | **0.033** | **0.043** | **[+0.0025, +0.0189]** | 5/22/0 |
| PT3 | LaMP-2-news | Per-Task | acc | 0.8331 | 0.8320 | −0.0011 | 0.327 | 0.317 | [−0.0032, +0.0000] | 0/26/1 |
| R11 | LaMP-1 | One-LoRA FT | acc | 0.5500 | 0.5600 | +0.0100 | 0.566 | 0.564 | [−0.0200, +0.0500] | 2/97/1 |
| PT4 | LaMP-1 | Per-Task | acc | 0.5800 | 0.5800 | **0.0000** | 1.000 | 1.000 | [−0.0300, +0.0300] | 1/98/1 |
| R12 | LaMP-7 | One-LoRA FT | R-1 | 0.5693 | 0.5706 | +0.0013 | 0.733 | 0.709 | [−0.0062, +0.0091] | 13/78/9 |
| PT5 | LaMP-7 | Per-Task | R-1 | 0.5649 | 0.5648 | −0.0001 | 0.979 | 0.925 | [−0.0062, +0.0061] | 7/86/7 |
| R13 | LaMP-2-movies | One-LoRA FT | acc | 0.8100 | 0.8200 | +0.0100 | 0.657 | 0.655 | [−0.0300, +0.0500] | 3/95/2 |
| PT6 | LaMP-2-movies | Per-Task | acc | 0.8000 | 0.8100 | +0.0100 | 0.657 | 0.655 | [−0.0300, +0.0500] | 3/95/2 |
| R14 | LaMP-5 | One-LoRA FT | R-1 | 0.5528 | 0.5554 | +0.0026 | 0.551 | 0.678 | [−0.0047, +0.0121] | 3/91/6 |
| PT7 | LaMP-5 | Per-Task | R-1 | 0.5653 | 0.5615 | −0.0037 | 0.562 | 0.679 | [−0.0168, +0.0085] | 8/82/10 |

**Nine of ten tracks are null.** Eight of ten point weakly positive, two weakly
negative; every CI except R10's spans zero.

### R10 is nominally significant but should NOT be read as a confirmed effect

R10 is the first User-LoRA round in this project to reach p<0.05 on both tests
with a CI excluding zero (R5 was at MDE p≈0.10; R6 p=0.20; R8 p=1.0; PT1
p=0.368; PT2 p=0.125). Three reasons to hold it loosely:

1. **Ten tracks were tested with no multiple-comparison correction.** At α=0.05
   across 10 tracks, ~0.5 false positives are expected by chance. A Bonferroni
   threshold would be 0.005; **R10's p=0.033 does not survive it.** This is the
   dominant caveat.
2. **Significance rides on 5 movers, not on magnitude.** 22 of 27 users are
   exact ties, so the Wilcoxon runs on 5 non-zero pairs that all happen to point
   the same way. A sign test on 5 same-direction pairs floors at p=0.031 — p≈0.04
   is near the smallest value this design *can* produce.
3. **The effect is ~1 accuracy point** on a 15-way task, from K=27, the smallest
   pool in the program.

### The dominant finding: the User-LoRA barely changes the model's output at all

Across all 1,354 paired records, the stacked per-user adapter changed the
prediction on **only 7.9%**:

| track | records | predictions changed | rate |
|---|---|---|---|
| LaMP-2-news r10 | 277 | 7 | 2.5% |
| LaMP-2-news pt3 | 277 | 3 | 1.1% |
| LaMP-1 r11 | 100 | 3 | 3.0% |
| LaMP-1 pt4 | 100 | 2 | 2.0% |
| LaMP-7 r12 | 100 | 28 | 28.0% |
| LaMP-7 pt5 | 100 | 20 | 20.0% |
| LaMP-2-movies r13 | 100 | 6 | 6.0% |
| LaMP-2-movies pt6 | 100 | 5 | 5.0% |
| LaMP-5 r14 | 100 | 14 | 14.0% |
| LaMP-5 pt7 | 100 | 19 | 19.0% |
| **TOTAL** | **1,354** | **107** | **7.9%** |

The classification tasks sit at 1-6%; the two generation tasks (LaMP-7, LaMP-5)
at 14-28%. So these nulls are not "personalization hurts" — they are mostly
"the adapter is barely perturbing the output distribution." On a closed-vocab
task where the answer is one token already strongly determined by the BM25
context, an r=8 q+v LoRA at LR=1e-5 for 3 epochs on a few hundred examples moves
almost nothing. That is a property of the recipe (deliberately carried over
unchanged from R5/OPPU), not of these tasks — and it bounds how large any
per-task effect could have been detected at these K.

### PT4 is a second exact 0.0000 cancellation

PT4 (LaMP-1 on Per-Task-LoRA) produced mean_diff = exactly 0.0000 with
1 win and 1 loss — structurally the same pattern R8 hit on LaMP-3 (5 wins /
5 losses, Δ=0.0000). Two independent occurrences of an exact cancellation now
exist; both are on classification tasks with very high tie rates, which is the
mechanical explanation (few movers, symmetric split), not a coincidence needing
a deeper account.

### Base-adapter swap: still no single story

The R-vs-PT comparison now has data on five more tasks, and it does not
consolidate:

- LaMP-3: One-LoRA FT base **destroyed** the lift (R8 = exact 0.000) while the
  Per-Task base **preserved a weak one** (PT1 ΔMAE −0.030).
- LaMP-2-news: **the opposite** — One-LoRA FT base is the positive one (R10),
  Per-Task base is flat/slightly negative (PT3).
- LaMP-4: R6 weak-positive, PT2 weak-negative (sign flip).
- LaMP-1 / LaMP-7 / LaMP-2-movies / LaMP-5: both bases null, no separation.

Six distinct base-swap shapes across seven tasks. The "task-mixing costs
personalization" hypothesis that R7/R8 pointed toward is not supported as a
general rule — it holds on LaMP-3 and reverses on LaMP-2-news.

### LaMP-7's thin corpus is a live confound, as flagged pre-execution

LaMP-7's pool floor is 14 profile entries (max 137) and its training signal is
unsupervised raw tweets. Its double null (R12 +0.0013, PT5 −0.0001) is therefore
weak evidence about the unsupervised recipe itself — low per-user token volume
is not ruled out as the cause. LaMP-1, on the same unsupervised path but with
158-533 entries per user, is also null, which is the more informative of the two.

## Conclusion

Personalization does not produce a robust, reproducible lift on any of these
five tasks under the unchanged OPPU recipe. The one nominally significant result
(R10, LaMP-2-news on One-LoRA FT) does not survive correction for the ten tracks
tested and rests on 5 movers out of 27 users; it is a lead worth a dedicated,
adequately-powered replication, not a finding.

The more actionable result is mechanical: **the per-user adapter changes the
prediction on only 7.9% of records**, and 1-6% on the classification tasks. Any
future round on these tasks should treat "make the adapter actually move the
output" as the prerequisite problem — a recipe question (rank, LR, epochs,
target modules) rather than another K=100 sweep at the current settings.

Q4 remains LaMP-3-specific, as R6 already established for LaMP-4.

## Provenance

- Pools: `data/lamp_user_stats/{LaMP_2_news_top27,LaMP_1_top100,LaMP_7_top100,LaMP_2_movies_top100,LaMP_5_top100}_users.json`
- T3 sizing: `data/lamp_user_stats/<task>_t3_sizing.json` — LaMP-1 (1792) and
  LaMP-5 (2048) carry `max_seq_length_override: true`; both had exactly one
  example above the 8192 positional ceiling (14,925 / 15,544 tokens) against
  next-largest per-user maxima of 1,597 / 2,048, so pinning the ceiling would
  have preserved nothing while costing headroom. `n_over_pinned: 1` each
  (0.0044% / 0.0039%).
- Adapters: `train/checkpoints/user_lora_<tag>_<user>_<track>_seed0/final`
- Consolidated predictions: `results/<task>_test_<track>_C{2,3}.predictions.jsonl`
- Paired comparisons: `results/paired_compare_c2_<track>_bm25_vs_c3_<track>_userlora_bm25*.json`

### Execution notes

- One transient `CUDA error: unspecified launch failure` (LaMP-2-movies R13,
  user u00006843) — retried via `condor/eval_lamp_user_lamp2movies_r13_retry1.sub`,
  no code change. Everything else ran clean on first submit.
- The 5 expected refuse-to-overwrite failures in the corpus-build batches were
  the smoke users recurring in the full queue, as predicted.
- All 12 + 8 eval smokes verified with `n > 0` before their batches — the PT1
  `LAMP_DIR` failure mode did not recur.
- `profile_stable` came back **True for all 27** LaMP-2-news users, so the
  softened profile-drift handling added during scaffolding never fired. The
  original LaMP-4 shared-index fast path was valid here too; that change was
  insurance, not a fix.
