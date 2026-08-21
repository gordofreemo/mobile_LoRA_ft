# h13 — on-device OPPU movie-tagging validation: build + verification log

Companion to `2026-08-19-ondevice-oppu-movie-validation-h13-plan.md`. Records the
pre-flight verification, the two bugs the build surfaced, and the campaign state.

## Pre-flight verification (the plan's five "verify on conduit BEFORE building" items)

1. **What the cluster movie OPPU arm stacks.** `run_oppu.py` does
   `PeftModel.from_pretrained(base, task_lora)` then `merge_and_unload()`, so each
   User-LoRA trains over a MERGED movie task adapter
   (`train/checkpoints/oppu_rep/movie_tagging/task_lora_k1`, r=8 α=8, q/k/v — its
   `out_proj` target is dead on SmolLM3). The device base is therefore a 4-bit MLX
   quantisation of THAT merge, built by `scripts/h13/fuse_oppu_movie_task.py`.
   **Verified exactly**: the merge reproduces the cluster run's recorded
   `merged_base_hash` `20cc0133e8d1ebf83c80b802e1edc240`.
2. **Loss masking + training text.** Prompt-prefix masking by token count:
   `full = tok(full_prompt)+EOS`, `loss_start = len(tok(prompt))`, `loss_end = len(full)`.
   Rather than re-implement the text construction, `scripts/h13/dump_movie_texts.py`
   runs THEIR `utils.py` + `prompt.json` + vendored BM25 on conduit and dumps the
   per-user training and eval strings; the Mac builder only tokenises them.
   Prefix-property violations: **0/5,558**. Examples hitting the 2048 cutoff: **0**.
3. **R5-ablation adapter coverage.** All 100 movie users have `oppu_k1_r5_user000..099`
   (plus 100 hot-recipe ones). Full coverage, no gaps.
4. **Queue.** Frozen in `data/oppu_movie/h13_queue.json` before the first run, by
   descending predicted paired-queries-per-device-hour. Corpus: 5,558 profile
   entries / 3,302 test queries over 100 users; median 36 train examples and 22
   queries per user; max example 401 tokens (nothing near the 1024 cap).
5. **Scoring entry point.** `oppu_rep_score.py` reads `{id, output}` JSON per arm, so
   device predictions enter by being written in that shape. `eval/h13_score.py`
   replicates their LaMP_2M mapping for the in-loop layer and **reproduces the
   published cluster numbers exactly** on their own prediction files:
   rag 0.4933, oppu_r5 0.5697, Δ +0.0763.

## Bug 1 — `rope_theta` lost in the merge (SEVERE, silent, wider than h13)

transformers ≥5 writes rope settings ONLY into a nested `rope_parameters` block.
Both `mlx-lm` (python) and **mlx-swift** read the TOP-LEVEL `rope_theta` and fall
back to **10000** when absent. SmolLM3 needs **5,000,000** — a 500× error that
leaves the model fluent but measurably worse.

Measured on 413 queries (first 15 users), same prompts throughout:

| plane | rag acc | invalid-label rate |
|---|---|---|
| cluster reference (bf16, HF) | 0.4939 | 0.000 |
| MLX bf16, `rope_theta` broken | 0.3535 | 0.000 |
| MLX 4-bit, `rope_theta` broken | ~0.32–0.36 | **0.187** |
| MLX bf16, fixed | **0.4939** | 0.000 |
| MLX 4-bit, fixed | 0.4383 | 0.000 |

Fixed bf16 MLX matches the cluster arm to four decimals, which validates the whole
h13 pipeline (prompts, tokenisation, merge, LoRA scale, scorer) end to end. The
18.7% degenerate "list every tag" outputs were entirely this bug.

**This is not h13-only.** `ageyko/SmolLM3-3B-a1lamp-4bit` on the Hub — the model the
device loads for h5/E2E, h7–h11 and the whole NAX-ON rerun campaign — has the same
missing key, so those rounds ran with `rope_theta = 10000`. **No published number
changes**: those rounds measured wall time, memory, energy and thermals, and MLX
kernel timings are value-independent; the NAX A/B compared two arms under the same
config. Only absolute loss values are affected. Any future *quality* claim over that
model must fix the config first. `mlx-community/SmolLM3-3B-4bit` is unaffected.

## Bug 2 — `loraScale` convention was inconsistent across rounds

`LoRALinear` computes `y + scale·(x@a)@b`, so `scale` IS `alpha/r`, exactly PEFT's
`lora_alpha/r`. h12 encoded this correctly (`2.0` for α8/r4). The h5-era constant
sets `loraScale = 16.0` for α16/r8, where the correct value is `2.0` — an 8×
over-scaling of the adapter output. Again this does not move any published h5–h11
number (they are cost measurements), but h13 is a quality claim, so it uses **2.0**,
matching the cluster adapters' `lora_alpha/r = 16/8`.

## Queue composition caveat (correcting the plan)

The plan anticipated the early prefix being biased toward *small*-profile users. The
frozen cost model does the opposite: fixed per-user costs (the 222 s cost-law
intercept, model loads) amortise over big users, so the queue starts with the
largest. Those are also the highest-effect users. On the cluster's own predictions,
the effect accumulated along this exact queue order runs:

| prefix | 1 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| queries | 452 | 726 | 890 | 1193 | 1622 | 2476 | 3302 |
| cluster Δ | +0.283 | +0.291 | +0.242 | +0.182 | +0.137 | +0.097 | +0.076 |

**Any prefix result overstates the full-pool effect and must be reported with this
table beside it.** The ordering rule was frozen blind to results, so this is a
disclosure item, not a selection problem.

## Declared deviations

* **Sampler.** MLX applies filters top_p → min_p → top_k and scales by temperature
  *after* filtering; HF applies temperature → top_k → top_p. At T=0.1 both are
  effectively greedy, and all four h13 arms share the identical MLX sampler, so the
  on-device comparison is internally consistent. Cross-plane comparison against the
  published bf16/HF numbers is **not** licensed.
* **Batching.** Batch-1 microbatches × 8 accumulation instead of their per-device
  2 × accum 4; token-weighted window normalisation makes the gradient equivalent.
* **Shuffle.** Their `Dataset.shuffle()` is unseeded, so byte-order parity with the
  cluster is unobtainable. The builder bakes a seeded per-epoch permutation, which
  is what makes the device-vs-Mac loss comparison a fidelity check.
* 4-bit base (not bf16), no LoRA dropout — as planned.

## Device fidelity (smoke, user 8000865)

Device and Mac consume the identical corpus file. Step-1 loss (pristine adapter, so
a pure forward-pass comparison): **device 0.52491 vs Mac 0.53191**, 1.3% relative —
A19 NAX 4-bit kernels vs M3's. All 12 queries: device predictions identical to both
the Mac-control and cluster adapters; rag 0.667 → all three adapter arms 0.750.

## Operational trap discovered (cost ~1 h)

`devicectl device copy to` with a **single** `--source` RENAMES that source to
`--destination`. `--source config.json --destination Documents/h13_model/` replaced
the `h13_model` DIRECTORY with a file, and a later single-file push replaced
`Documents` ITSELF with an 18-byte file; every subsequent read returned
`CoreDeviceError 7000`, and neither killing `remotepairingd` nor a device reboot
touched it. Recovery: push a PARENT DIRECTORY as `--source <stage> --destination
Documents`. Multiple `--source` arguments place items inside the destination; a lone
one renames. Encoded in `scripts/h13/run_h13_user.sh`.

## Build inventory

* `scripts/h13/dump_movie_texts.py` — their-code text dump (runs on conduit)
* `scripts/h13/build_h13_device_data.py` — pre-tokenised per-user corpora + eval prompts
* `scripts/h13/build_queue.py` — frozen queue
* `scripts/h13/fuse_oppu_movie_task.py` — task-adapter merge (hash-verified, config-hardened)
* `scripts/h13/convert_peft_adapter_to_mlx.py` — cluster adapters → MLX
* `scripts/h13/diag_eval_plane.py` — the bf16/4-bit × rag/cluster diagnostic above
* `train/train_user_mlx_h13.py` — Mac control arm
* `eval/h13_score.py` — reporting kit
* `scripts/h13/run_h13_user.sh`, `run_h13_queue.sh` — device sequencers
* `ios/.../Benchmark/LLMEvaluator+H13.swift` — device train + four-arm on-device eval

## First queue user (rank 0, `8000201`, 452 queries) — 2026-08-19

Device cost: train 94.8 min (291 steps, predicted 90), 5 min cooldown, 22 min to
evaluate 452 queries × 3 arms in one model load (0.97 s per query-arm). Loss
0.31 → 0.066. Total 122 min. The `mac` arm was skipped — Mac-control training is
paused during the day, and `run_h13_mac_catchup.sh` fills it in later.

| arm | accuracy |
|---|---|
| cluster reference, RAG (bf16, HF) | 0.5155 |
| cluster reference, OPPU r5 (bf16, HF) | 0.7987 |
| **device: rag** (4-bit MLX, on phone) | **0.2854** |
| **device: cluster adapter** | **0.7677** |
| **device: device-trained adapter** | **0.7588** |

**The effect survives on-device training.** `device − rag` = **+0.4634** query-level
(t_p 2.8e-50, 238/203/23 over both completed users), and the device-trained adapter
lands on the cluster-trained one: `device − cluster` = −0.0086, ns (p=0.35), with
only 5.6% of predictions differing between them. That is the h13 claim.

**But the on-device Δ is larger than the cluster's (+0.463 vs +0.283) for a reason
that must be reported, not celebrated: the baseline degrades, not the adapter.**
Under 4-bit, the adapter arm loses 0.03 (0.799 → 0.768) while the RAG arm loses
0.23 (0.5155 → 0.285). This user's gold is 75% "based on a book"; the bf16 RAG arm
emits that tag 169/452 times, the 4-bit RAG arm only 107/452. So quantisation
damages *in-context* personalization far more than *weight-baked* personalization —
a coherent and interesting story, but on n=1 user so far, and one whose RAG arm
leans hard on a single retrieved example. The earlier 15-user diagnostic put the
4-bit RAG penalty at only −0.056 (0.4939 → 0.4383), so the size of this penalty is
strongly user-dependent.

**Open check for the overnight window:** run the Mac 4-bit rag arm on this user to
confirm 0.285 is the quantised model's honest score and not a device-side artefact.
The two planes agreed exactly on the 15-user diagnostic, so this is a confirmation,
not a suspicion.

⚠ n=2 users / 464 queries. The grouped per-user test is meaningless at this size
(n=2), and the queue front-loads the highest-effect users — see the prefix table
above.

## Prefix 10 (10 users, 1,152 queries) — 2026-08-20

| arm | accuracy | invalid |
|---|---|---|
| rag | 0.3177 | 0.001 |
| cluster adapter | 0.6233 | 0.007 |
| **device-trained adapter** | **0.6293** | 0.003 |

| contrast | query mean | t_p | W/T/L | grouped | grouped t_p | changed |
|---|---|---|---|---|---|---|
| cluster − rag | +0.3056 | 7.9e-77 | 384/736/32 | +0.1349 | 0.087 | 0.442 |
| **device − rag** | **+0.3116** | 2.0e-78 | 392/727/33 | +0.1464 | 0.064 | 0.444 |
| device − cluster | +0.0061 | 0.307 | 27/1105/20 | +0.0115 | 0.045 | 0.084 |

**The claim holds and is tracking the reference.** As the queue moves past its
high-effect head, the on-device Δ decays in the same shape as the cluster's own:
device−rag goes +0.425 (prefix 5) → +0.312 (prefix 10) against the cluster's
+0.242 → +0.182 over the same prefixes. The device arm stays level with the
cluster arm throughout (0.6293 vs 0.6233).

**Do not read the grouped `device − cluster` p = 0.045 as an effect.** It is one
of five contrasts at n=10 users, reported at every prefix, with no correction —
exactly the shape R10 and warm-start's LaMP-5 had, and both were discarded. The
query-level test on the same contrast is null (p = 0.31, 27 wins / 20 losses out
of 1,152), and only 8.4% of predictions differ between the two adapters.

Device pace: ~15–19 min per user at this size (train + 5 min cooldown + eval).

## Prefix 20 (20 users, 1,599 queries) — 2026-08-20

| arm | accuracy | invalid |
|---|---|---|
| rag | 0.3483 | 0.001 |
| cluster adapter | 0.5716 | 0.005 |
| **device-trained adapter** | **0.5822** | 0.002 |

| contrast | query mean | t_p | W/T/L | grouped | grouped t_p | changed |
|---|---|---|---|---|---|---|
| cluster − rag | +0.2233 | 2.7e-72 | 396/1164/39 | +0.0733 | 0.064 | 0.343 |
| **device − rag** | **+0.2339** | 1.9e-77 | 411/1151/37 | +0.0907 | 0.023 | 0.343 |
| device − cluster | +0.0106 | 0.041 | 43/1530/26 | +0.0174 | 0.0039 | 0.087 |

Still tracking the reference as the queue decays: device−rag +0.425 → +0.312 →
+0.234 over prefixes 5/10/20, against the cluster's own +0.242 → +0.182 → +0.137.

### Watch item: the device adapter may be beating the cluster adapter

`device − cluster` has been positive and strengthening at every prefix —
+0.0061 (p=0.31) at 10, **+0.0106 (p=0.041), grouped +0.0174 (p=0.0039)** at 20,
with the device arm ahead on the raw accuracy too (0.5822 vs 0.5716). The
manipulation check says the arms genuinely differ on only 8.7% of predictions,
and among those the device wins 43 to 26.

**There is a plausible mechanism, which is why this is worth watching rather than
dismissing: the device adapter is trained on the same 4-bit weights it is
evaluated over, while the cluster adapter was trained on bf16 and is being ported
onto a 4-bit base.** A quantisation-aware adapter beating a ported full-precision
one is exactly what QLoRA-style reasoning predicts, and it would be a directly
relevant result for an on-device paper — "train the adapter where you deploy it".

**But treat it as a hypothesis, not a finding, until the queue completes.** It is
one of five contrasts, computed at every prefix, uncorrected — the shape that
produced R10 and warm-start's LaMP-5, both discarded. The whole effect is ~17
queries out of 1,599. If it survives to n=100 with the mechanism intact, it is
worth a dedicated arm (the same user trained on bf16 vs 4-bit, evaluated on 4-bit);
if it decays, it was multiplicity.

## FINAL — 98 of 100 queue users, 3,277 paired queries (2026-08-21)

Two users remain (`8000324` and the tail of the queue); the phone locked at 16:48
and the runner is waiting for an unlock. Nothing below will move materially.

### The claim holds

| plane | baseline | adapter | Δ query-level |
|---|---|---|---|
| cluster reference (bf16, HF) | 0.4910 | 0.5679 | +0.0769 |
| **on-device (4-bit MLX, phone)** | **0.3634** | **0.4843** | **+0.1208** |

All four numbers are over the same 98 users / 3,277 queries. `device − rag` is
**+0.1208** query-level (t_p 9.7e-71, 458/2757/62) and **+0.0265 grouped per-user
(t_p 0.0043)**.

**The grouped per-user effect transfers almost exactly: +0.0265 on-device against
+0.0236 for the cluster reference on the same users.** That is the cleanest
statement of the result — the personalization effect OPPU measures with a
bf16 GPU pipeline survives end-to-end on-device training, evaluated under 4-bit
deployment conditions, at the same per-user magnitude.

### Quantisation hits retrieval harder than weights

| arm | bf16 → 4-bit |
|---|---|
| RAG baseline | **−0.1276** |
| personalized adapter | **−0.0836** |

**In-context personalization degrades 1.53× harder than weight-baked
personalization.** This is why the on-device query-level Δ (+0.121) exceeds the
cluster's (+0.077): the baseline falls further than the adapter arm does. Reported
as an asymmetry, never as "personalization works better on-device".

### The device-trained adapter matches the cluster-trained one

`device − cluster` = +0.0067 query-level (p=0.061), grouped +0.0038 (p=0.33), with
the arms differing on only 8.4% of predictions (80 wins / 58 losses). Training the
adapter on the phone costs nothing measurable against training it on an A100.

### The watch item was multiplicity, and it is retracted

`device − cluster` looked like a real effect mid-campaign and is not:

| prefix | query mean | p | grouped | grouped p |
|---|---|---|---|---|
| 10 | +0.0061 | 0.31 | +0.0115 | 0.045 |
| 20 | +0.0106 | 0.041 | +0.0174 | 0.0039 |
| 49 | +0.0123 | 0.0030 | +0.0161 | 0.0014 |
| 80 | +0.0070 | 0.065 | +0.0037 | 0.373 |
| **98** | **+0.0067** | **0.061** | **+0.0038** | **0.333** |

It rose monotonically across three prefixes, reached grouped p = 0.0014, and then
decayed to null. The proposed mechanism (the device adapter is trained on the same
4-bit weights it is evaluated over, the cluster adapter is ported from bf16) was
plausible and still is — it simply is not supported here. **No follow-up arm is
warranted.**

Keep this as a methodology exhibit: peeking at prefixes of an anytime queue
manufactures exactly the R10 / warm-start-LaMP-5 shape, and only the full run
settles it.

### Cost

~46 h wall-clock for 98 users including two multi-hour interruptions when the
phone was unplugged; roughly 26 h of active device time. Per user: 2–95 min of
training (median ~4 min), a 5 min cooldown, and ~1 s per query-arm of evaluation.
Peak memory 2.22 GiB. Every arm generated on the phone.
