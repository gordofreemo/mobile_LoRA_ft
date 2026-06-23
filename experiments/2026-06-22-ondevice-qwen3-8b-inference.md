# On-device inference characterization — Qwen3-8B-4bit on iPhone 17 Pro

**Date:** 2026-06-22 · **Phase 3** · descriptive characterization (no hypothesis test)
**Design:** reuses the pre-registered SmolLM3-3B plan
`experiments/2026-06-21-ondevice-base-inference-plan.md` **verbatim** (same star
grid, same device-state regime, same harness), with only the subject model
swapped to a 7B-class model. No separate plan file (no hypothesis/gate to
pre-register — this is exploratory perf telemetry).
**Harness:** `app_build` `qwen3-8b-ondevice-bench-h3` (h3 = per-run
`GPU.resetPeakMemory()` so `peak_mem_bytes` is a clean **per-cell** peak instead
of a session high-water mark, + `git_commit`/`git_dirty` baked into every
record). Source vendored in-repo via git subtree at
`ios/mlx-swift-examples/Applications/LLMEval/Benchmark/`.

## Hypothesis

None — exploratory. The goal was simply to get the same on-device perf telemetry
(decode/prefill tok/s curves, TTFT, cold start, peak memory, thermal behavior)
for a **7B-class** model as the 2026-06-21 SmolLM3-3B-4bit characterization, to
have a second size point on record and to answer the implicit feasibility
question: *does the next size class up even fit and run on the phone?*

**Subject model = `mlx-community/Qwen3-8B-4bit`** (8.2B params — the natural
"next size up"; SmolLM3 has no 7B variant, so this is a different architecture
and tokenizer, not within-family size scaling). Chosen because it is first-class
in `mlx-swift-lm` (registry arch `qwen3`, loads with no custom handler) and has
an `enable_thinking` toggle, so the thinking-off regime matches the SmolLM3
benchmark exactly.

## Setup

- **Subject:** `mlx-community/Qwen3-8B-4bit` (base, no adapter), MLX-Swift `LLMEval`.
  ~4.7 GB 4-bit weights, downloaded on-device from HF on first generation (one-time,
  Wi-Fi; coexists with the cached SmolLM3-3B-4bit — separate Hub repo dirs).
- **Device:** iPhone 17 Pro (`iPhone18,1`), iOS 26.5.1.
- **Device state:** USB-connected + plugged + **airplane mode ON** (enabled *after*
  the model download) + Low Power Mode OFF + battery ~100%, unlocked, idle-timer
  disabled. `battery_level / charging / low_power_mode / thermal_state` logged per run.
- **Regime:** thinking OFF (`enable_thinking:false`, verified — realistic-tier
  answers are a single rating digit, no `<think>` leak). Greedy (temperature 0).
- **Forced fixed length:** EOS suppressed by driving `TokenIterator` directly until
  `maxTokens` (same mechanism as the 3B run).
- **Grid (star, identical to the 3B run):** prefill prompt∈{64,256,512,1024,2048}@gen 64;
  decode gen∈{128,512,1024,2048}@prompt 256; 5 measured + 1 discarded warmup per cell;
  realistic tier n=5 natural-EOS; stress = forced decode to 300 s / per-256-token segments.
- **Memory instrumentation (h3):** `GPU.resetPeakMemory()` before each measured
  generation → `peak_mem_bytes` is now the per-cell peak (weights resident + prefill
  KV + decode activations), the fix flagged in the 3B writeup.

Launch (Mac-driven, single `--benchmark` = `.full` mode runs the **entire** suite —
cold + prefill + decode + realistic + stress — in one session):

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark
# pull + aggregate:
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl --destination /tmp/devpull/bench_metrics.jsonl
grep qwen3-8b-ondevice-bench-h3 /tmp/devpull/bench_metrics.jsonl \
  > results/ondevice/bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl
python eval/bench_aggregate.py results/ondevice/bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl \
  --out results/ondevice_base_qwen3_8b_4bit_2026-06-22.json
```

**Single-session run (no split needed):** unlike the 3B (whose `--benchmark`
overran a `--console` session and needed a `--benchmark-tail` follow-up), the full
8B grid completed in one launch — but it ran **~95 minutes** (decode-2048 alone is
~5 tok/s × 2048 tok × 5 runs ≈ 33 min), so almost everything after the first
prefill cell ran on a thoroughly heat-soaked device (see caveat). Telemetry:
`results/ondevice/bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl` (68 records).
Aggregate: `results/ondevice_base_qwen3_8b_4bit_2026-06-22.json`.

## Result

5 measured + 1 discarded warmup per cell. **Thermal context dominates this run** —
see the caveat below; only the cold record and the prefill-64 cell are `nominal`.

### Headline: feasibility — YES, it fits

**Qwen3-8B-4bit runs on the iPhone 17 Pro with no OOM/jetsam.** Clean per-cell peak
(h3): **4.74 GB** at a 64-token prompt → **5.43 GB** at 2048 tokens (KV growth now
visible per-cell). Well under the cap granted by the app's
`com.apple.developer.kernel.increased-memory-limit` entitlement on this 12 GB
device. The model loads from the on-device cache and generates correctly.

### Cold start (n=1, `nominal` — the first measurement, on a cool device)
- **Model load: 2043 ms** (1.5× the 3B's 1362 ms — 2.7× the weights).
- **Cold TTFT: 770 ms** (2× the 3B's 380 ms).
- → **app-launch-to-first-answer ≈ 2.8 s** (vs the 3B's ~1.7 s). Peak 4.95 GB.
- Cold variance (the 3B's 3× `--benchmark-cold` relaunches) was **not** collected:
  after a 95-min session the device is deeply heat-soaked, so further launches
  would be hot-device cold starts, not comparable. This single cold sample is the
  cleanest possible (cool, `nominal`, first thing measured). n=1 — same status as
  the original 3B anecdote before its variance pass.

### Prefill curve (gen fixed 64) — *device heats across the sweep*
| prompt_tok | gen tok/s | prefill tok/s | TTFT (ms) | peak_mem (GB) | thermal |
|---:|---:|---:|---:|---:|---|
| 64   | 15.5 ± 0.5 | 239 | 268    | 4.74 | **nominal** |
| 256  | 15.8 ± 0.2 | 323 | 797    | 5.01 | fair |
| 512  | 15.5 ± 0.2 | 329 | 1566   | 5.22 | fair |
| 1024 | 15.1 ± 0.3 | 341 | 3029   | 5.26 | fair |
| 2048 | 13.7 ± 1.0 | 322 | 6433   | 5.43 | serious |

Decode rate is ~15.5 tok/s and fairly flat (the modest 15.5→13.7 decline conflates
KV growth with the device heating into `serious` by the 2048 cell). Prefill
throughput **240–340 tok/s**. TTFT scales ~linearly with prompt length, 268 ms →
6.4 s. Peak memory climbs smoothly with the resident KV cache (clean per-cell, h3).

### Realistic tier (LaMP-3-shaped prompt, natural EOS, n=5 — **hot device**)
- **~12 tok/s** (first run 8.6, then 12.2–12.3), clean **1-token** answers (a rating
  digit → confirms thinking-off works for Qwen3), TTFT ~1.8 s, peak 4.95 GB, all
  `serious`. This is the deployment-shape number but on an already-throttled device.

### Decode-length curve (prompt fixed 256) — **fully thermally confounded (`serious`)**
| gen_tok | gen tok/s | single-gen time | thermal |
|---:|---:|---:|---|
| 128  | 8.9–15.6 (degrades within cell) | 8–14 s | serious |
| 512  | ~6.1 | ~84 s | serious |
| 1024 | ~5.6 | ~183 s | serious |
| 2048 | 4.7–5.4 | up to **434 s** | serious |

By the time the decode sweep ran (after the multi-minute prefill sweep), the device
was heat-soaked and never recovered while plugged (cooldowns hit the 120 s cap each
time, never returning to `nominal`). These are **hot-device, throttled** rates, not
steady-state. decode-128 visibly degrades *across its own 5 repeats* (15.6 → 8.9)
as the device continues heating — the cleanest in-cell picture of the throttle.

### Sustained stress (forced decode, prompt 256, ~300 s)
- **7 segments, 1792 tokens, 7.7 → 6.1 tok/s (−21%).** The decay looks *milder* than
  the 3B's −53% only because the device entered the stress cell **already `serious`**
  (the 3B entered its stress cell cool); the 8B had already done its throttling
  earlier in the session.
- `thermalState` reported **`serious`** throughout — note this is the **opposite** of
  the 3B finding, where the enum stayed a useless `nominal` while throughput halved.
  At 8B the sustained load is heavy enough that the coarse enum *does* catch it. The
  per-segment tok/s signal (decision 6) remains the reliable channel regardless.

### 3B vs 8B — clean cross-model channel (cold + prefill-64, both `nominal`)
| metric | SmolLM3-3B-4bit | Qwen3-8B-4bit | ratio |
|---|---:|---:|---:|
| decode tok/s (p64, nominal) | 38.8 | 15.5 | 0.40× |
| prefill tok/s (p64) | 684 | 239 | 0.35× |
| cold model-load (ms) | 1362 | 2043 | 1.50× |
| cold TTFT (ms) | 380 | 770 | 2.0× |
| app-launch→first-answer (s) | ~1.7 | ~2.8 | 1.6× |
| peak memory | ~2.2 GB | ~5.0 GB | ~2.3× |

The ~2.5× decode and ~2.9× prefill slowdowns track the ~2.7× parameter ratio
(decode is memory-bandwidth-bound, prefill compute-bound — both scale ~linearly
with params at fixed quantization). Memory scales ~2.3× (weights 2.8×; KV/activation
overhead differs by arch). **Different architecture/tokenizer**, so read this as
"Qwen3-8B vs SmolLM3-3B on the same silicon," not pure size scaling within a family.

## Conclusion

A 7B-class (8.2B) 4-bit model is **deployable** on the iPhone 17 Pro: it loads,
fits in **~5.0–5.4 GB** (no OOM under the increased-memory-limit entitlement), and
generates correctly. The cost vs the 3B deployment is roughly: **decode ~2.5×
slower (~15 vs ~37 tok/s), prefill ~2.9× slower, cold start ~2.8 s vs ~1.7 s,
memory ~2.3×** — all tracking the parameter ratio.

Two findings shape any 8B-class deployment decision:
1. **The 8B heat-soaks the phone fast and hard.** It reaches `serious` *within* the
   prefill sweep and sustained decode collapses to ~5 tok/s (a 2048-token answer
   can take >7 minutes). Plugged-and-idle, the device cannot recover between heavy
   cells. Any real 8B deployment is thermally bounded, not throughput-bounded.
2. **At 8B, `ProcessInfo.thermalState` actually reports the throttle** (`serious`),
   unlike the 3B where it lied `nominal`. Still trust per-segment tok/s as primary;
   but the enum is a usable coarse alarm at this load.

**Clean steady-state decode curves remain unobtainable while plugged** — even more
so than for the 3B. The pre-registered **unplugged-over-Wi-Fi follow-up** (3B plan
decision 7) is the way to get a clean 8B decode-length curve and cold-start
variance on a device that can return to `nominal`; **deferred, not blocking** for
this exploratory pass.

## Files & provenance (on disk)

| What | Path |
|---|---|
| **Raw telemetry** (68 records, one flat-JSON line per generation/segment) | `results/ondevice/bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl` |
| **Aggregate** (per-cell mean±std/min/max, cold, stress decay) | `results/ondevice_base_qwen3_8b_4bit_2026-06-22.json` |
| Aggregator (stdlib only, unchanged) | `eval/bench_aggregate.py` |
| Harness source (ours, h3) | `ios/mlx-swift-examples/Applications/LLMEval/Benchmark/{BenchmarkSupport,LLMEvaluator+Benchmark}.swift` + the `modelConfiguration` id in `LLMEvaluator.swift` |
| Pre-registered design (reused) | `experiments/2026-06-21-ondevice-base-inference-plan.md` |
| 3B comparison run | `experiments/2026-06-21-ondevice-base-inference.md` |

Records carry `git_commit` (`dc43e62`) + `git_dirty` (`true` — built from the working
tree carrying the h3 harness edits, committed immediately after this run) along with
`device_model`, `os_version`, `app_build`, `bench_session_id`, thermal/battery state,
`timestamp_utc`. Slice by `cell` ∈ {cold, prefill, decode, realistic, stress},
`warmup`, `forced`. Re-aggregate any time:
`python eval/bench_aggregate.py <jsonl> --out <path>`.
