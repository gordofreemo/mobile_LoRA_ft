# User-LoRA for the remaining 5 LaMP tasks — Execution Plan

**Status:** SCAFFOLDED 2026-07-30 — all code, pools, configs-generators and Condor subs are built and locally verified; nothing has been submitted (`condor_submit` stays user-run). See "§Scaffolding built 2026-07-30" at the end of this doc for what exists, what changed vs. the pinned design, and the submit order.
**Design pinned:** 2026-07-27, via `/grill_me`.
**Scope extended:** 2026-07-30 — rounds now cover **all five** remaining tasks, not three (see §Scope extension).

This is the one design doc covering three separate, interdependent pieces of
work, grilled together because the user asked to parallelize the whole
"get User-LoRA working on all 7 LaMP tasks" effort in one session rather than
sequencing task-by-task.

---

## Why this doc exists

As of 2026-07-26, User-LoRA (per-user personalization) numbers exist for
only 2 of the 7 LaMP tasks: LaMP-3 (R5, R8, PT1) and LaMP-4 (R6; R9 planned,
PT2 built this session — mechanical, no design needed, see below). The other
5 tasks (LaMP-2-movies, LaMP-2-news, LaMP-5, LaMP-1, LaMP-7) have zero
User-LoRA rounds. The 2026-07-25 correction
([[project-user-lora-per-task-viability]]) established none of them are true
dead ends, but left concrete open questions this doc resolves:

1. **LaMP-2-movies and LaMP-5** are viable via the same profile-entry
   reframing trick already implemented for LaMP-3/4, but `build_user_dataset.py`
   has never been extended to cover them.
2. **LaMP-2-news** is viable via real per-user *records* (not reframing —
   it has genuine per-user training-record volume, unlike movies/5), but no
   round has been designed for it, and its natural pool (K~27) is much
   smaller than every other round's K=100.
3. **LaMP-1 and LaMP-7** are genuinely history-misaligned per the OPPU paper
   (arXiv:2402.04401) and need an entirely new training method — unsupervised
   next-token prediction on right-shifted raw history — that doesn't exist
   in this codebase.

**PT2 (LaMP-4 User-LoRA on Per-Task-LoRA) is NOT part of this doc.** It was
built directly in this session without a grill because it required zero
design decisions — pure mechanical replication of PT1's pattern onto R6's
existing K=100 pool. Scaffolding (template config, generator, train/eval
subs, aggregator) is already committed-pending; see the PT2 section of
`project_per_task_lora_pt1_design.md` memory for its naming chain. Not yet
smoke-tested or submitted — condor_submit stays user-run.

---

## Piece 1 — `build_user_dataset.py` reframing extension (LaMP-2-movies, LaMP-2-news, LaMP-5)

### Pinned design

| # | Axis | Decision |
|---|---|---|
| 1 | Scope | All three tasks in one code change, not just movies/5 — LaMP-2-news needs a `TASKS` entry too (for BM25 system-context formatting), even though its framing differs from movies/5. |
| 2 | LaMP-2-movies framing | **`profile`** framing (record-level dead end — every unseen user has exactly 1 record total, split across train/dev/test; the real training volume is in profile entries, avg 124/user, max 774, verified via the actual time-split data). New `PROFILE_FRAMING["LaMP_2_movies"] = ("description", "tag")`. |
| 3 | LaMP-2-movies `wrap_user_text` | Synthesizes the full eval-shaped question from a profile entry: `f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [{', '.join(shuffled_labels)}] description: {trim(entry['description'])}"`. Verified byte-for-byte against 1,410 real `LaMP_2_movies` questions (100% prefix match). |
| 4 | LaMP-2-movies label bracket order | **Per-example randomized order**, seeded deterministically (e.g. `random.Random((task, entry_id)).shuffle(labels)` — implementation detail, not a design fork). Real eval questions have arbitrary per-example label order (verified — doesn't match our own alphabetical `LAMP2_MOVIES_LABELS` constant, and there's no way to replicate the *specific* real shuffle since it's baked into the static LaMP dataset with no exposed formula). Training with one fixed order risks the model learning a positional shortcut that doesn't hold at eval time; per-example randomization is the closer structural match to real eval variance. |
| 5 | LaMP-2-news framing | **`records`** framing — real per-user record volume exists (top user: 211 train / 36 dev / 35 test), so the existing `--framing records` path (real `input`/gold from `train_outputs.json`, byte-exact, no synthesis) applies directly, same as R6 used for LaMP-4. **No bracket-ordering problem** — sidesteps Q3/Q4's issue entirely since the real `input` string (with its own real bracket order) is used verbatim. Needs a `TASKS["LaMP_2_news"]` entry only (index_field/format for BM25 system-context lines), no `PROFILE_FRAMING` entry. |
| 6 | LaMP-5 framing | **`profile`** framing (record-level dead end, same as movies — avg profile 89/user, max 533). New `PROFILE_FRAMING["LaMP_5"] = ("abstract", "title")`. |
| 7 | LaMP-5 `wrap_user_text` | `f"Generate a title for the following abstract of a paper: {entry['abstract']}"` — fixed prefix, verified byte-for-byte against all 1,500 real `LaMP_5` dev questions (100% match, single space after the colon; apparent extra spaces in some examples come from the abstract text's own leading whitespace, not the wrapper). No per-example variation needed — this is a plain generation task, not classification. |
| 8 | `TASKS` entries (index_field/format) for all three | Mechanically duplicated from `eval_lamp.py`'s existing `TASKS` dict (same values already used for BM25 system-context formatting at both train and eval time) — `LAMP2_MOVIES_LABELS`/`LAMP2_NEWS_LABELS` constants also duplicated for the bracket-shuffle step. |

**Outcome:** after this change, `build_user_dataset.py --task LaMP_2_movies --bm25-k 4 --user <fp>`, `--task LaMP_2_news --framing records --bm25-k 4 --user <fp>`, and `--task LaMP_5 --bm25-k 4 --user <fp>` all work. No round-specific scaffolding yet — that's Piece 2 (news) and Piece 3 (movies/5 are covered by a future PT-round, not designed here since no round was requested for them this session; only LaMP-2-news, LaMP-1, and LaMP-7 get rounds in this doc).

---

## Piece 2 — LaMP-2-news User-LoRA round (R10 + PT3, parallel)

### Pinned design

| # | Axis | Decision |
|---|---|---|
| 1 | Task-LoRA base(s) | **Both One-LoRA FT and Per-Task-LoRA(LaMP-2-news) in parallel** — A1-lamp never trained on this task, so unlike LaMP-3/4 there's no "original round to re-run later." Runs as both the next R-number (**R10**) and the next PT-number (**PT3**) simultaneously, sharing the same pool/data, differing only in `base_adapter`. |
| 2 | Pool / K | **K=27**, threshold `n_train≥4 AND n_dev≥4` among unseen users. This is the natural maximal pool — the qualifying set's actual minimum `n_train` is 18 (a real gap in the distribution between 4 and 17), so there's no arbitrary thin-user problem despite the loose-looking threshold. All 27 have ≥3 test records (avg 10.3, max 35). |
| 3 | Framing | `--framing records --bm25-k 4` (per Piece 1 #5) — real per-user train records, same mechanism as R6's LaMP-4. |
| 4 | Recipe | Unchanged OPPU recipe (r=8 q_proj+v_proj, α=16, dropout=0.05, LR=1e-5, wd=1e-2, 3 epochs, cosine + 3% warmup, per_device_batch=2, grad_accum=4) — not relitigated, same as every prior round. `max_seq_length` needs its own T3 sizing (mechanical measurement, like `LaMP_4_round6_t3_sizing.json`) — not copy-pasted from another task. |
| 5 | Eval structure | Per-user procs for both arms (baseline + stacked), same as R6/PT2 — NOT PT1's `--user-records-from-file` shortcut, since LaMP-2-news users have multiple test records each (3–35), and that shortcut only pulls one `test_record_id` per user. |
| 6 | Metric / comparison | Classification (accuracy + macro-F1, `score_classification`/`parse_closed_vocab_label`, already implemented in `eval_lamp.py`). Paired comparison needs a **new `--metric` flag on `eval/paired_compare_per_user.py`** (currently hardcoded to `rouge1`) — add `accuracy` using the same exact-match scorer `paired_compare.py` already has, default stays `rouge1` for R6/R9/PT2 backward compatibility. Macro-F1 stays a separate top-line descriptive, not folded into the per-record paired framework. |
| 7 | Statistical framing | No pre-registered gate (same convention as R6/R8/R9/PT1/PT2) — descriptive reporting only. |
| 8 | Round numbering | **R10** (base = One-LoRA FT), **PT3** (base = Per-Task-LoRA(LaMP-2-news)). Both share pool/data; only `base_adapter` and output paths differ, same delta pattern PT1 applied to R5→PT1 and this session applied to R6→PT2. |

**Not yet done, needed before implementation:** T3 sizing script run (mechanical), pool-selection script (`select_top_users`-style, K=27 threshold), per-user training corpora built for all 27 users (`--framing records`), config templates + generators for both R10 and PT3, train/eval/aggregate/paired-compare subs for both tracks (8 sub files total, following PT2's just-built pattern).

---

## Piece 3 — LaMP-1 / LaMP-7 unsupervised right-shifted-history training (new method)

OPPU's paper (Section 3, arXiv:2402.04401v3) describes this path in one
sentence with no equation, no chunking/formatting spec, no loss-masking
discussion, and no task-specific results for LaMP-1/LaMP-7: *"we replace the
user history output y_u in personal PEFT training objectives with
right-shifted history x_u' for unsupervised next token prediction."* Every
concrete implementation choice below was made in this session, not sourced
from the paper — flagged explicitly since this piece is qualitatively more
speculative than everything else in this doc.

### Pinned design

| # | Axis | Decision |
|---|---|---|
| 1 | What this collapses to | Plain causal-LM fine-tuning on raw history text — no chat template, no system/user/assistant roles, no assistant-span masking, loss on every token. Structurally simple; the BM25-retrieval-into-system-message mechanism used everywhere else in this codebase does **not** apply to training here (it stays unchanged at eval time). |
| 2 | Training example granularity | **One example per profile entry** — each tweet (LaMP-7) or each `"{title}\n\n{abstract}"` (LaMP-1), raw text, no wrapper — matching the existing profile-entry-per-example convention (LaMP-3/4/5), not a new per-user-concatenated-corpus format. |
| 3 | Code architecture | **New script, `train/train_unsupervised_clm.py`** — not a mode bolted onto `train.py`. Reuses the same provenance/config/LoRA-setup/`base_adapter`-stacking machinery, but its own `build_example`-equivalent (no chat template, no masking) and its own JSONL corpus format (`{"text": ...}`, not `{system, user, assistant}`). Keeps `train.py`'s single well-tested SFT path unrisked by every other round depending on it — matches this project's existing convention of duplicating rather than deep-branching a shared script (`build_user_dataset.py`'s own stated rationale). |
| 4 | Task-LoRA base(s) | **One-LoRA FT and Per-Task-LoRA(<task>) in parallel for both LaMP-1 and LaMP-7** (dual R/PT track, same pattern as Piece 2). A1-lamp is skipped even though it's technically viable for LaMP-7 (A1-lamp's original mix was LaMP-{3,4,7}) — kept consistent with every other new round this session rather than reopening A1-lamp as a special case. |
| 5 | LaMP-1 profile-snapshot sourcing | **Falls back to the user's own TEST record's `profile` field** when no train-split record exists for that user. Necessary, not optional: every one of LaMP-1's 9,542 unseen users appears in *exactly one* split (6,542 train-only, 1,500 dev-only, 1,500 test-only) — **zero** users have both a train record and a test record, so the existing train-record-only snapshot lookup (`find_latest_train_record`) yields zero eligible users. Not a leakage concern: the `profile` field on any record (train/dev/test) is the user's history *prior to* that record, never the record's own query/gold — the same field `eval_lamp.py` already reads for BM25 retrieval at eval time, and profile entries already do double duty as training data + retrieval context everywhere else in this project (LaMP-3/4/5's reframing works the same way). |
| 6 | LaMP-7 profile-snapshot sourcing | Unchanged — standard train-record lookup already works (all 331 eligible LaMP-7 users have both a train record and a test record). |
| 7 | Pool / K | **K=100 for both tasks**, matching every other round. LaMP-1: 1,500 eligible users (test-record-sourced profiles, avg 85 entries, min 47) — comfortable headroom. LaMP-7: 331 eligible users, avg ~16 profile entries/user — thinner than LaMP-1 but comparable to LaMP-4's low end (which R6 already ran successfully at K=100). |
| 8 | Recipe | Unchanged OPPU hyperparameters (same as Piece 2 #4) — the OPPU paper itself uses identical settings across supervised and unsupervised paths (Table 5), no distinction. `max_seq_length` needs its own T3 sizing per task. |
| 9 | Eval structure | **Unchanged from every other round** — eval methodology (BM25 retrieval + supervised prompt shape at inference) is untouched; only the User-LoRA's *training* objective differs. Both LaMP-1 and LaMP-7 have exactly 1 test record per eligible user (verified), so eval uses the same flat, efficient pattern as LaMP-3/PT1: `eval/paired_compare.py` (not the grouped per-user version) and the single-job `--user-records-from-file` baseline shortcut. |
| 10 | Metric | LaMP-1: classification, `--metric accuracy`. LaMP-7: generation, `--metric rouge1` (both already supported by the flat `paired_compare.py`, no further script changes needed beyond Piece 2 #6's `paired_compare_per_user.py` extension, which isn't even used here). |
| 11 | Round numbering | **R11 + PT4** = LaMP-1. **R12 + PT5** = LaMP-7. (Order is arbitrary — task-number order, not a priority statement.) |
| 12 | Statistical framing | No pre-registered gate, descriptive only — same convention as every round since R6. |
| 13 | Framing as exploratory | This piece's numbers should be reported with an explicit caveat that the training methodology itself (not just the personalization result) is a first attempt at an underspecified recipe from the source paper — distinct in kind from Pieces 1–2, which are mechanical replications of already-validated patterns. |

**Not yet done, needed before implementation:** `train/train_unsupervised_clm.py` itself (new script), `build_user_dataset.py`'s snapshot-lookup fallback for LaMP-1 (Piece 3 #5), per-entry unsupervised corpus builder (new `--framing unsupervised` or equivalent in `build_user_dataset.py`, or a new sibling script — not yet decided at the file level, only the objective is pinned), T3 sizing for both tasks, pool-selection scripts, config templates + generators for 4 tracks (R11/PT4/R12/PT5), and all associated subs.

---

---

## Scope extension (2026-07-30) — LaMP-2-movies and LaMP-5 get rounds too

As pinned, this doc designed rounds for only 3 of the 5 tasks; LaMP-2-movies
and LaMP-5 were left "code-viable, round not designed" (see §What's
explicitly NOT covered). Checking the actual user tables while implementing
Piece 1 showed the two deferred tasks have **exactly the same structural
problem as LaMP-1**, whose fix was already pinned here in Piece 3 #5:

| task | unseen users | with a test record | with **both** train & test |
|---|---|---|---|
| LaMP-1 | 9,542 | 1,500 | **0** |
| LaMP-2-movies | 8,040 | 1,557 | **0** |
| LaMP-5 | 17,682 | 1,500 | **0** |
| LaMP-2-news | 102 | 102 | 102 |
| LaMP-7 | 2,994 | 331 | 331 |

So the test-record profile-snapshot fallback that LaMP-1 needs is exactly
what LaMP-2-movies and LaMP-5 need, and with Piece 1's reframing on top,
neither requires any new design decision. User confirmed extending scope
rather than deferring them to a later session.

**Round numbering continues the same two tracks:** R13 + PT6 = LaMP-2-movies,
R14 + PT7 = LaMP-5, both K=100, supervised profile-entry reframing, same
unchanged OPPU recipe. That makes **ten tracks across five tasks**, and closes
User-LoRA coverage on all 7 LaMP tasks.

---

## Corrections to the pinned design, found against real data (2026-07-30)

Two pinned decisions turned out to rest on wrong premises. Both were checked
directly against the time-split data before changing anything.

**1. Piece 1 #4 (LaMP-2-movies label bracket order) — REVERSED.** The pinned
decision was a deterministic *per-example shuffle* of the 15 tags, on the
premise that real eval questions carry "arbitrary per-example label order."
They do not. Across **all 1,410 dev records plus the first 1,500 test and
1,500 train records (4,410 records), there is exactly ONE tag order**:

```
[sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy,
 classic, psychology, fantasy, romance, thought-provoking, social commentary,
 violence, true story]
```

What the original check had actually established was only that this order
isn't *alphabetical* (i.e. differs from `LAMP2_MOVIES_LABELS`) — not that it
varies. Since the order is fixed and observable, and `eval_lamp.py` feeds the
record's real `input` through unchanged at inference, reproducing it verbatim
is what the cardinal train/eval-consistency rule demands; shuffling would have
introduced a gratuitous train/eval mismatch. Implemented as a new
`LAMP2_MOVIES_PROMPT_ORDER` constant with an assertion that it's a permutation
of the label universe.

**2. Question-text trimming/stripping — DROPPED for the two new profile-framing
tasks.** The pinned wrapper for LaMP-2-movies used `trim(description)` and the
natural reading for LaMP-5 was `abstract.strip()`. Measured against 400 real
dev questions each, both break byte parity: `trim` truncates at 600 chars and
collapses whitespace (rewriting 18/400 movie questions), and `.strip()`
destroys leading whitespace on 7/400 and trailing on 49/400 LaMP-5 abstracts,
all of which the real `input` preserves. Both now interpolate the raw field.
After the fix, reconstructing a real `input` from its own body through
`wrap_user_text` is **400/400 byte-identical for both tasks**, with LaMP-3
unchanged at 50/50 (regression check).

**3. Records-framing profile-stability assertion — SOFTENED to a strategy
choice.** `emit_record_bm25` used to hard-fail if a user's train records
didn't all carry an identical profile (an invariant verified for LaMP-4, but
never tested against LaMP-2-news, whose users hold up to 211 train records).
It now checks the fingerprint and *picks the strategy*: one shared BM25 index
when stable (LaMP-4's fast path, unchanged), else one index per record over
that record's own profile — which is the semantically correct thing anyway,
since a record's profile is its own history-prior snapshot. Which path ran is
recorded as `profile_stable` in the meta sidecar, so it can never be a silent
difference.

**4. `--metric accuracy` is parse-then-match, not exact-match.** The pinned
text said to reuse "the same exact-match scorer `paired_compare.py` already
has." That would be wrong for the classification tasks: `pred` in a
predictions JSONL is the model's **raw generated text**, and `eval_lamp.py`
runs it through a per-task parse function before comparing to gold. A strict
exact-match scorer would mark every prose-wrapped prediction ("The category is
politics.") wrong and would not aggregate to the accuracy `eval_lamp.py`
reported for the same file. Both `paired_compare.py` and
`paired_compare_per_user.py` now mirror `eval_lamp.py`'s parsers for LaMP-1 /
LaMP-2-movies / LaMP-2-news, and fall back to strict equality elsewhere — so
LaMP-3's existing behaviour is untouched.

---

## Scaffolding built 2026-07-30

Nothing submitted. All of the following is on disk and locally verified.

### Code
| file | what changed |
|---|---|
| `train/build_user_dataset.py` | Piece 1: `TASKS`/`PROFILE_FRAMING` entries for all 7 tasks; `wrap_user_text` for LaMP-2-movies + LaMP-5; `LAMP2_MOVIES_PROMPT_ORDER`; test/dev profile-snapshot fallback (`resolve_snapshot`) with a self-record drop guard; new `--framing unsupervised`; source-field emptiness check |
| `train/train_unsupervised_clm.py` | **new** — OPPU right-shifted-history trainer: no chat template, no role mask, loss on every token, labels = input_ids (the shift is the model's own), same config/provenance/`base_adapter` machinery as `train.py` |
| `eval/paired_compare_per_user.py` | `--metric {rouge1,accuracy}` (parse-then-match) + `--round-tag`; defaults preserve R6/R9/PT2 filenames byte-identically |
| `eval/paired_compare.py` | `accuracy_scorer(task)` now parse-then-match for the closed-vocab tasks; LaMP-3 unchanged |
| `eval/aggregate_user_predictions_newtask.py` | **new** — handles both eval patterns, picked from the pool JSON; same gold byte-match leakage check as R5/R6/PT1 |
| `data/select_top_users_newtasks.py` | **new** — generic pool selector, per-task eligibility + K |
| `data/lamp_user_stats/newtask_t3_sizing.py` | **new** — generic T3 sizing; tokenizes the way the matching trainer will (chat template vs raw) |
| `data/lamp_user_stats/newtask_gen_configs.py` | **new** — emits the 10 per-track templates + per-user configs; refuses to run without that task's T3 JSON |
| `condor/gen_newtask_subs.py` | **new** — emits all 78 sub files with the tyr1/modi exclusions, Blackwell capability range, and `LAMP_DIR` override baked in |

### Pools (already built and on disk)
| task | K | profile size range | eval pattern |
|---|---|---|---|
| LaMP-2-news | 27 (= the whole qualifying pool) | 37–423 | grouped (3–35 test records/user) |
| LaMP-1 | 100 | 158–533 | flat (1 test record/user) |
| LaMP-2-movies | 100 | 150–774 | flat |
| LaMP-5 | 100 | 177–521 | flat |
| LaMP-7 | 100 | **14**–137 | flat |

**LaMP-7's floor of 14 profile entries is thin** — thinner than any prior
round's — and its corpus is unsupervised raw tweets. Worth watching: if R12/PT5
comes back as a flat null, low per-user token volume is a live confound, not
just a negative result.

### Submit order (each step gated on the previous; `condor_submit` stays user-run)

Per task, in this order — the five tasks are independent and can run in parallel:

1. `condor/build_user_dataset_<tag>_smoke.sub` → check the corpus JSONL looks right
2. `condor/build_user_dataset_<tag>.sub` (K procs; one expected refuse-to-overwrite failure on the smoke user)
3. `condor/t3_sizing_<tag>.sub` → pins `max_seq_length`; **stops non-zero if anything exceeds 8192**
4. `python data/lamp_user_stats/newtask_gen_configs.py --task <task>` (login node, instant)
5. per track (R and PT run in parallel):
   `train_user_lora_<tag>_<track>_smoke.sub` → `..._<track>.sub`
   → `eval_lamp_user_<tag>_<track>_smoke.sub` (**check record COUNTS, not just exit codes**)
   → `eval_lamp_user_<tag>_<track>_baseline.sub` (flat tasks only) → `eval_lamp_user_<tag>_<track>.sub`
6. `python eval/aggregate_user_predictions_newtask.py --task <task> --track <track>`
7. `condor/paired_compare_<tag>_<track>.sub`

Tags: `lamp2news`, `lamp1`, `lamp7`, `lamp2movies`, `lamp5`.
Tracks: `r10`/`pt3`, `r11`/`pt4`, `r12`/`pt5`, `r13`/`pt6`, `r14`/`pt7`.

**Cost note:** 854 per-user LoRA trainings (2×27 + 8×100) plus their evals. This
is a much larger batch than any prior round; the K=100-everywhere choice is the
pinned one, but it is worth a deliberate look before submitting all five tasks
at once rather than staggering them.

---

## What's explicitly NOT covered by this doc

- PT2 (LaMP-4 on Per-Task-LoRA) — already built, mechanical, no grill needed (see "Why this doc exists").
- ~~User-LoRA for LaMP-2-movies and LaMP-5~~ — **superseded 2026-07-30**: both now have full rounds (R13/PT6, R14/PT7), see §Scope extension.
- Any change to eval methodology, BM25 retrieval, the Task-LoRA training recipes, or any existing round's results.

## Sequence

1. Implement Piece 1 (`build_user_dataset.py` extension) — prerequisite for Piece 2.
2. Implement Piece 2 (LaMP-2-news R10 + PT3) — independent of Piece 3.
3. Implement Piece 3 (`train_unsupervised_clm.py` + LaMP-1/LaMP-7 R11/PT4/R12/PT5) — independent of Piece 2, can run in parallel with it.
4. Smoke-test each piece before its full K-sized batch (project convention, no exceptions).
5. `condor_submit` stays user-run throughout, per [[feedback-user-owns-commits-submits]].
