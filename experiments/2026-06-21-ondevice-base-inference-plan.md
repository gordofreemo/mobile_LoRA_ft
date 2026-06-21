# On-device base-inference benchmark — design plan (Phase 3)

**Status:** PLAN (pre-registered design, pre-implementation). Locked 2026-06-21
via /grill_me. Implementation not yet started.

**Subject:** `mlx-community/SmolLM3-3B-4bit` (base model, no adapter) running
inference on the testbed **iPhone 17 Pro** (`iPhone18,1`, iOS 26.5.1) via the
MLX-Swift `LLMEval` app. See the Phase 3 runbook in `CLAUDE.md` for the
build/deploy/pull mechanics.

## Hypothesis

This is **descriptive characterization, not a hypothesis test.** Goal: produce a
reusable, reproducible on-device performance baseline for the base 4-bit model
that becomes the comparison floor for everything Phase 3 measures next (base vs.
Task-LoRA on-device, and later on-device *training* metrics). No paired/
inferential statistics in this pass — those belong to the later base-vs-LoRA
comparison.

## Why now / context

As of 2026-06-21 there is exactly **one** on-device generation record on the
phone (`Documents/bench_metrics.jsonl`): gen 37.5 tok/s, prompt 438 tok/s, TTFT
144 ms, peak mem 1.82 GB, 63 prompt / 399 gen tokens, thinking off. n=1 is an
anecdote — this plan turns it into a characterization with error bars and
scaling curves.

## Design decisions (locked, with rationale)

Each was resolved one-by-one; the rationale is kept so a future session does not
relitigate.

1. **Goal = reusable characterization baseline** (not a one-off milestone
   snapshot). Build the measurement rig once; reuse it verbatim for the
   Task-LoRA and on-device-training measurements. Stage it: small warm number
   first (validates harness end-to-end), then the full grid.

2. **Harness = in-app benchmark loop, auto-started by a launch arg, fully
   Mac-driven.** No per-generation human taps. `devicectl device process launch
   --console --terminate-existing` passes a launch arg (verified: devicectl
   supports `command-line-arguments` + `-e/--environment-variables`); the app
   detects the flag, runs the sweep on launch instead of waiting for a UI tap,
   disables the idle timer for the duration, writes each result to
   `bench_metrics.jsonl`, and exits. `--console` blocks until the app
   terminates → clean done-signal → then `devicectl copy from` pulls the JSONL.
   The entire run is **one CLI command** (satisfies the project's "launchable
   from one CLI command" reproducibility rule). Only manual step: one Face-ID
   unlock at the start (a locked phone won't foreground the app for GPU work).

3. **Generation length = forced fixed length for the grid** (suppress EOS so the
   model emits exactly N tokens; content is irrelevant to per-token decode cost)
   **+ a few natural-EOS runs** for realism. Forcing makes the gen-length axis
   exact and the decode curve clean; natural-EOS carries the real-world number.

4. **Prompt-length axis = synthetic exact-length padding** at **64 / 256 / 512 /
   1024 / 2048** tokens, **binned by the measured `prompt_tokens`** (not the
   target). Prefill cost is ~content-independent, so synthetic padding gives the
   cleanest controllable x-axis. Cap 2048 (real LaMP prompts never approach it).
   The natural-EOS realistic runs instead use a **real LaMP-3 BM25-k4-in-system
   prompt** (the true deployment input shape).

5. **Grid = star / one-axis-at-a-time, NOT full cross-product.** Decode tok/s is
   mostly a function of KV size = prompt_len + tokens_so_far, so a Cartesian
   product just re-measures the same effect and self-throttles.
   - **Prefill curve:** prompt ∈ {64,256,512,1024,2048} @ fixed gen 64.
   - **Decode curve:** gen ∈ {128,512,1024,2048} @ fixed prompt 256.
   - **Realistic tier:** ~5 natural-EOS runs on the real LaMP-3 prompt.
   - **5 measured + 1 discarded warmup per cell.**

6. **Thermal strategy.** Two distinct phenomena handled differently:
   - *Inter-cell comparability:* log `thermal_state` every run; cooldown between
     cells is **gated on `thermalState == nominal`** (cap the wait ~120 s, record
     the actual wait), not a blind sleep. Primary numbers come from
     nominal-start runs; fair/serious runs are flagged, never silently averaged.
   - *Intra-run sustained throttle:* one dedicated **~5-min sustained stress
     run** (forced continuous generation) logging **per-segment tok/s** (e.g.
     every 256 tokens) → tps-vs-time decay curve. This run is *expected* to leave
     nominal; that's the point. (The 2048-token decode cell only captures ~55 s
     of sustained decode — not enough for the real throttle signal.)

7. **Device state (reproducibility knobs).** USB-connected + **plugged** +
   **airplane mode ON** (after the one-time model download) + **Low Power Mode
   OFF** + brightness fixed low + battery > 50%. Plugged adds a small charging-
   heat offset, which is *conservative* (true unplugged perf ≥ measured) and buys
   a maximally reliable one-command pipeline (no flaky wireless pairing, no
   network interrupts). `battery_level / charging / low_power_mode /
   thermal_state` are all logged so it stays interpretable. Unplugged-over-Wi-Fi
   is a clean follow-up refinement, not a first-pass requirement.

8. **Cold-start = first-class metric, separate from steady-state.** The first
   generation after launch bundles three separable costs: weight load (~1.73 GB
   disk→unified mem), first-token Metal-shader compile, and steady-state. Capture
   it: one cold generation first on launch, recorded `cold=true` with
   `model_load_ms` (bracket the lazy container load) and its inflated `ttft_ms` —
   this *is* the "app launch → first answer" number. Then run the warm grid
   (model resident; each cell still discards its own warmup for per-config
   compile). **Never average cold into steady-state.** For cold-start variance,
   do ~3 **separate tiny launches** (launch → 1 gen → exit), each `devicectl
   launch` being one cold sample — cheap, isolates the reboot/disk-cache-
   sensitive number.

9. **Regime = thinking-off throughout.** SmolLM3 defaults thinking ON, but the
   whole project lives thinking-off: the Task-LoRA was trained with the chat
   template applied thinking-off, all prior LaMP eval is thinking-off, and
   train/eval consistency is the project's cardinal rule. To keep the eventual
   base-vs-Task-LoRA comparison apples-to-apples, the base must be measured
   thinking-off. Thinking-on is not a hardware fact anyway (forced-length decode
   tok/s is identical — same kernel; thinking only changes *natural* length, a
   model-behavior fact in a regime we never ship). Thinking-on dropped from the
   headline entirely.

## Schema (flat single-level JSON, one record per generation)

Repo convention = flat scalar records so `pd.DataFrame([...])` works with zero
unnesting.

- **Existing (keep):** `model, prompt_tokens, gen_tokens, ttft_ms, gen_tps,
  prompt_tps, gen_time_s, peak_mem_bytes, max_tokens, thinking, truncated,
  device_model, os_version, timestamp_utc`.
- **Add — device/thermal state (per run):** `thermal_state`
  (nominal/fair/serious/critical), `battery_level`, `charging`,
  `low_power_mode`.
- **Add — run semantics:** `cold` (bool), `model_load_ms`, `warmup` (bool),
  `forced` (bool — EOS suppressed vs natural), `target_prompt_tokens`,
  `target_gen_tokens` (requested values; measured `prompt_tokens`/`gen_tokens`
  remain the ground truth you bin on).
- **Add — labeling/provenance:** `bench_session_id` (one UUID per `devicectl
  launch`), `cell` (`prefill|decode|realistic|cold|stress`), `repeat_idx`,
  `app_build` (build-time string baked into the app), `bench_schema_version`.
- **Stress run = one flat record per segment** (`cell="stress"`, `segment_idx`,
  `cumulative_tokens`, `gen_tps` = that segment's rate). No nested arrays — keeps
  the flat-JSON-to-DataFrame rule; the decay curve is a filter+plot on
  `cell=="stress"`.

## Setup (the run, once implemented)

Mac-driven, one command (exact form to be finalized at implementation):

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C \
  mlx.LLMEvalJGW9U9Y36Y --benchmark
# then pull:
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl \
  --destination results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl
```

Phone prepped per decision 7 (plugged, airplane on, LPM off, brightness low,
battery > 50%, unlocked).

## Analysis & outputs

- **Aggregator:** tracked `eval/bench_aggregate.py` (mirrors `eval/summary.py`).
  Reads the pulled JSONL → per-cell **mean ± std (n), min/max** of `gen_tps /
  prompt_tps / ttft_ms / peak_mem_bytes`, after filtering `warmup==true` and
  segregating `cold` and serious/critical-thermal runs out of steady-state.
  Cold-start gets its own summary line; the stress run is a `cell=="stress"`
  filter → tps-vs-`cumulative_tokens` decay series. Pure pandas groupby.
- **Stats depth:** descriptive only (mean ± std + n + min/max). **No bootstrap
  CIs / paired tests** — reserved for the base-vs-Task-LoRA comparison.
- **Where results land:**
  - Raw telemetry → `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl`
  - Aggregated flat JSON → `results/ondevice_base_smollm3_4bit_2026-06-21.json`
  - Experiment log → `experiments/2026-06-21-ondevice-base-inference.md`
    (Hypothesis/Setup/Result/Conclusion, with the verbatim launch command).

## Provenance loose end — MUST fix as part of this work

The iOS app (`ios/mlx-swift-examples/`) is gitignored, so harness edits are
untracked and these numbers would otherwise be unreproducible (violating
reproducibility-first). Fix: **check the benchmark harness into a tracked path**
(a `.patch` against `LLMEvaluator.swift`, or a tracked copy under
`ios/patches/`), and **bake a build-time `app_build` string into the app** so
every JSONL ties back to a specific harness version. This closes the runbook's
own standing TODO.

## Implementation order

1. **Swift harness first** (nothing can be captured until it exists): benchmark
   loop in `LLMEvaluator.swift` + launch-arg auto-start + idle-timer disable +
   schema additions + `app_build`. Rebuild/redeploy per the runbook
   (`-skipMacroValidation` required).
2. **`eval/bench_aggregate.py`** aggregator.
3. Run the grid (one CLI command), pull, aggregate, write the results experiment
   log.
4. Check in the harness `.patch` (provenance fix).

### Two implementation-time verifications (not design decisions)

Confirm against `mlx-swift-lm` source, not blogs:
- **How to suppress EOS** for forced fixed-length generation (likely set the
  EOS-token logit to `-inf`, or a max-only stop condition in the generate loop).
- **Where the lazy container load exposes a completion point** to bracket for
  `model_load_ms`.

---

*Design locked 2026-06-21 (commit 350cee0) via /grill_me. Implementation
pending.*
