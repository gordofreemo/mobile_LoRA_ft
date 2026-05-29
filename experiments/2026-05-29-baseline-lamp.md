# 2026-05-29 — Baseline zero-shot LaMP

- **Branch:** master
- **Commit (eval state):** `dfa54361d1a02415aab2e49df61f7ab35b718f22`
- **Predecessor commits:** `ba174ba` (Add LaMP baseline zero-shot eval + cluster infra), `1ce9912` (Project skeleton)
- **Container:** `ghcr.io/gordofreemo/smollm3-train:ver2` (Condor docker universe, 1 GPU)
- **Host:** `gerda.hpc.uni-saarland.de` (per result `hostname`)

---

## Hypothesis

This establishes the **Baseline** in CLAUDE.md's comparison chain
`Baseline → A1-lamp → A1-full → A2`: the bare SmolLM3-3B with no LoRA adapter on the
LaMP dev splits. Two sub-conditions are run, answering different questions:

1. **Profile baseline** (BM25 k=4 retrieved profile in the system prompt) — the direct
   control every later adapter is measured against. Same harness, same prompt format, same
   decoding, same seed across all conditions, so any later delta is attributable to the
   adapter, not the eval setup.
2. **Non-personalized floor** (no profile, just the task input) — answers *does the LaMP
   user profile help at all?*. If floor ≈ profile baseline, retrieval is doing nothing and
   the whole personalization premise is in trouble before we ever train.

---

## Setup

### What ran (files in the repo)

| Role | File |
|---|---|
| Eval harness | `eval/eval_lamp.py` — BM25 retrieve → chat-template prompt (thinking off) → greedy decode → parse → metrics → flat result JSON + predictions JSONL |
| Profile baseline submit | `condor/eval_lamp.sub` — `arguments = --task $(task) --split dev --k 4 --seed 0`; `queue task in (LaMP_3 LaMP_4 LaMP_7)` |
| Floor submit | `condor/eval_lamp_floor.sub` — same harness, `--no-profile` |
| Interactive smoke | `condor/interactive.sub` (GPU shell), `python eval/eval_lamp.py ... --limit 5` |
| Summary table | `eval/summary.py` — globs `results/*.json` → one-row-per-run table |
| Approach / methodology rationale | `notebooks/lamp_evaluation_approach.md` (gitignored, personal) |
| Implementation walkthrough | `notebooks/learning/03_evaluating_on_lamp.md` (gitignored, personal) |

LaMP-6 is intentionally absent — the public release ships only Avocado file-id
placeholders (no email text), so it can't be scored without the licensed corpus.

### Model & data

- Model: `data/models/SmolLM3-3B` (HuggingFaceTB/SmolLM3-3B, bf16, frozen).
- Data: `data/lamp/LaMP_{3,4,7}/dev_questions.json` + `dev_outputs.json`.
  Sizes: LaMP-3 = 2500, LaMP-4 = 1925, LaMP-7 = 1500. Question/gold id alignment verified
  (100% match in each task).

### Decoding

- `do_sample=False` (greedy) — deterministic.
- `enable_thinking=False` via the chat template; verified in smoke output (no `<think>`
  blocks) and by `parse_fail_rate=0.0004` on 2500 LaMP-3 outputs.
- `max_new_tokens`: 8 (LaMP-3), 64 (LaMP-4 / LaMP-7).
- Seeds: `random`, `torch`, `cuda` all set to `0`.

### Conditions

| Condition | Retriever | k | Adapter |
|---|---|---|---|
| Profile baseline | BM25 (pure-Python, in-script) | 4 | none |
| Floor (`--no-profile`) | — | — | none |

### Reproduction commands

From project root, with HEAD at `dfa5436`:

```bash
# Smoke verification (interactive GPU session)
condor_submit -i condor/interactive.sub
cd /home/ange00008/projects/mobileFT_distill
python eval/eval_lamp.py --task LaMP_3 --split dev --k 4 --limit 5
python eval/eval_lamp.py --task LaMP_4 --split dev --k 4 --limit 3
python eval/eval_lamp.py --task LaMP_7 --split dev --k 4 --limit 3
exit

# Full runs (3 parallel jobs each)
condor_submit condor/eval_lamp.sub          # profile baseline
condor_submit condor/eval_lamp_floor.sub    # floor

# Read the table
python3 eval/summary.py
```

### Provenance captured per run

Each result record stores: `timestamp_utc`, `hostname`, `command`, `seed`, `python_version`,
`torch_version`, `transformers_version`, `peft_version`. **`git_commit` came back `null`**:
the Docker image doesn't include `git`, so `subprocess(["git", "rev-parse", "HEAD"])` failed
inside the container and the field fell through to `None`. Code state for this run is
pinned to HEAD `dfa5436` (recorded here, in this entry, as the closest reproducible
reference). Fix for the next runs: install `git` in the image OR pass the SHA in via
Condor `environment = GIT_COMMIT=...` and have `collect_provenance()` read the env var as
a fallback.

---

## Result

| Task | Cond | n | Metric | MAE | parse-fail | prompt-tok | gen-tok | s/ex |
|---|---|---|---|---|---|---|---|---|
| LaMP_3 | floor | 2500 | acc **0.4404** | 0.628 | 0.0000 | 258.6 | 2.0 | 0.06 |
| LaMP_3 | bm25k4 | 2500 | acc **0.6924** | 0.369 | 0.0004 | 741.2 | 2.4 | 0.10 |
| LaMP_4 | floor | 1925 | rouge1 **0.1410** | — | — | 105.3 | 26.3 | 1.07 |
| LaMP_4 | bm25k4 | 1925 | rouge1 **0.1615** | — | — | 340.6 | 24.8 | 1.01 |
| LaMP_7 | floor | 1500 | rouge1 **0.4139** | — | — | 107.6 | 28.2 | 0.78 |
| LaMP_7 | bm25k4 | 1500 | rouge1 **0.4352** | — | — | 226.3 | 27.6 | 1.89 |

Lift (floor → profile baseline):

| Task | Δ metric | Relative | Δ prompt tok |
|---|---|---|---|
| LaMP_3 | **+0.252 acc** (MAE halves) | +57 % | +482 |
| LaMP_4 | +0.020 rouge1 | +15 % | +235 |
| LaMP_7 | +0.021 rouge1 | +5 % | +118 |

**Output artifacts**

- Flat result records (one row per run, plot-ready):
  `results/LaMP_{3,4,7}_dev_base_{bm25k4,noprofile}_seed0.json`
- Per-example predictions (one `{id, pred, gold}` per line):
  `results/LaMP_{3,4,7}_dev_base_{bm25k4,noprofile}_seed0.predictions.jsonl`
- Job stdout/stderr: `runlogs/eval_lamp{,_floor}.<task>.<ClusterId>.<ProcId>.{out,err}`

**Sanity checks**

- Question↔gold id match: 100 % across all three tasks (2500, 1925, 1500).
- LaMP-3 golds are exactly `{1,2,3,4,5}` strings.
- Parse failures: 1 / 5000 LaMP-3 outputs across both conditions → thinking is genuinely
  off and `parse_rating()` is robust.
- `.err` files contain only the `Loading weights` tqdm bar (stderr by default) — clean.

---

## Conclusion

1. **LaMP-3 carries the personalization story under retrieval-based prompting.** +25
   accuracy points and MAE halving is large; rating prediction is exactly where retrieved
   `(review, score)` history is most informative. The Task-LoRA will have to beat 0.69, so
   there is still ~30 points of headroom to perfect.
2. **LaMP-4 and LaMP-7 are nearly flat under retrieval** (~+2 ROUGE-1 each). Two
   complementary readings to keep in mind:
   - (a) BM25-retrieved past items don't transfer headline/tweet *style* well, only topic;
   - (b) ROUGE-1 is order- and synonym-blind, so it under-counts stylistic personalization
     that doesn't share surface tokens with the gold.

   Either way, this is the regime where the **synthetic-data Task-LoRA must earn its keep**.
   `A1-lamp → A1-full` (RQ2) is set up to catch exactly that.
3. **Retrieval's efficiency cost is heterogeneous** (+482 / +235 / +118 prompt tokens for
   LaMP-3/4/7). LaMP-3 buys its big lift at the highest per-query token cost — on-axis with
   the cluster-vs-on-device cost story Phase 2 measures.
4. **Per-example latency was 6× lower than the warmup-amortized smoke estimate**
   (0.06–1.89 s vs ≥0.6 s). Sharding the eval within a task remains unjustified.
5. **The baseline is reproducible.** Every later condition (`A1-lamp`, `A1-full`, `A2-*`)
   runs the same `eval/eval_lamp.py` harness with only `--adapter` changing, so deltas are
   attributable to the adapter, not the eval setup.

---

## Open / next

- Fix `git_commit` capture (image lacks `git`) — install git in the image OR pass the SHA
  in via Condor `environment = GIT_COMMIT=$ENV(GIT_COMMIT)` and have
  `collect_provenance()` read the env var as a fallback.
- Commit `eval/summary.py` so the analysis path is reproducible from a tagged commit.
- BFCL regression on the bare model still owed (target ≥ 90, ref 92.3 per CLAUDE.md);
  separate harness `eval/eval_bfcl.py`, not yet written.
- Next milestone: **A1-lamp** Task-LoRA training (`train/train.py --condition a1_lamp`,
  not yet written). The numbers above are the bar to beat.
