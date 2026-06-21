# On-device base-inference characterization — SmolLM3-3B-4bit on iPhone 17 Pro

**Date:** 2026-06-21 · **Phase 3** · descriptive characterization (no hypothesis test)
**Design (pre-registered, locked):** `experiments/2026-06-21-ondevice-base-inference-plan.md`
**Harness:** `app_build` `smollm3-ondevice-bench-h1` (prefill/decode/cold) +
`smollm3-ondevice-bench-h2` (realistic/stress/cold) — identical measurement code,
h2 only adds the `--benchmark-tail` launch mode. Tracked at
`ios/patches/ondevice-benchmark-harness.patch`.

## Hypothesis

None — this turns the prior single on-device record (n=1: gen 37.5 tok/s, prompt
438 tok/s, TTFT 144 ms, peak 1.82 GB) into a reproducible characterization with
error bars + scaling curves, to serve as the comparison floor for base-vs-Task-LoRA
on-device and later on-device-training measurements.

## Setup

- **Subject:** `mlx-community/SmolLM3-3B-4bit` (base, no adapter), MLX-Swift `LLMEval`.
- **Device:** iPhone 17 Pro (`iPhone18,1`), iOS 26.5.1 (Build 23F81).
- **Device state (decision 7):** USB-connected + plugged + **airplane mode ON** +
  Low Power Mode OFF + brightness low + battery 100%, unlocked. `battery_level /
  charging / low_power_mode / thermal_state` logged per run.
- **Regime:** thinking OFF throughout (decision 9). Greedy (temperature 0).
- **Forced fixed length:** EOS suppressed by driving `TokenIterator` directly and
  ignoring the stop-token set until `maxTokens` (the EOS check lives in MLXLMCommon's
  loop wrapper, not in `TokenIterator.next()`).

Verbatim launch commands (Mac-driven, one command each; one Face-ID unlock):

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
# full grid (cold + prefill + decode):
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark
# tail (realistic + 5-min stress):
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark-tail
# cold-start variance (×3 separate launches):
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark-cold
# pull + aggregate:
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl \
  --destination results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl
python eval/bench_aggregate.py results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl \
  --out results/ondevice_base_smollm3_4bit_2026-06-21.json
```

**Run split (why two sessions):** the full grid was pre-registered as one `--benchmark`
command, but the nominal-gated inter-cell cooldowns (capped 120 s) plus the long
decode/stress cells overran a single `--console` session (~25 min cap hit during the
decode-2048 cell). The completed prefill + decode data was kept; realistic + stress
were captured in a follow-up `--benchmark-tail` launch (added in h2) rather than
re-running (and re-heating) the clean prefill/decode cells. Three extra
`--benchmark-cold` launches gave cold-start variance.

Telemetry: `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl`
(89 records incl. 1 legacy n=1 record, ignored by the aggregator — no `cell`).
Aggregate: `results/ondevice_base_smollm3_4bit_2026-06-21.json`.

## Result

5 measured + 1 discarded warmup per cell. Primary numbers = nominal-thermal runs.

### Cold start (n=5 separate launches)
- **Model load: 1362 ± 114 ms** (1.73 GB weights, disk → unified memory).
- **Cold TTFT: 380 ± 6 ms** (vs warm ~150–180 ms — the delta is first-token Metal
  shader compile). → **app-launch-to-first-answer ≈ 1.7 s.**

### Prefill curve (clean — all cells nominal; gen fixed 64)
| prompt_tok | gen tok/s | prefill tok/s | TTFT (ms) | peak_mem (MB)* |
|---:|---:|---:|---:|---:|
| 64   | 38.8 ± 0.1 | 684 ± 16 | 176 ± 4    | 1965 |
| 256  | 37.5 ± 0.4 | 731 ± 27 | 427 ± 16   | 2010 |
| 512  | 35.8 ± 1.1 | 743 ± 20 | 765 ± 20   | 2106 |
| 1024 | 34.5 ± 0.3 | 642 ± 28 | 1685 ± 70  | 2106 |
| 2048 | 32.0 ± 0.8 | 624 ± 19 | 3372 ± 102 | 2196 |

Decode rate declines smoothly 38.8 → 32.0 tok/s as the resident prompt (KV size)
grows 64 → 2048. Prefill throughput 624–743 tok/s. TTFT scales ~linearly with prompt
length (176 ms → 3.4 s); at the realistic LaMP-3 prompt size (~210 tok) it's well
under 400 ms.

### Realistic tier (real LaMP-3 BM25-k4 prompt, natural EOS, n=5)
- **35.0 ± 1.2 tok/s**, 1-token answers (a rating digit), TTFT 363 ± 1 ms,
  peak ~1.96 GB. This is the true deployment number for LaMP-3 rating prediction.

### Decode-length curve (prompt fixed 256) — **thermally confounded, see caveat**
| gen_tok | gen tok/s | n nominal | thermal |
|---:|---:|---:|---|
| 128  | 28.5 ± 0.1 | 2/5 | nominal+fair |
| 512  | 29.9 ± 8.1 | 2/5 | nominal+fair (high σ) |
| 1024 | 20.6 ± 1.9 | 0/5 | fair+serious |
| 2048 | 15.4 ± 0.4 | 0/4 | serious |

These cells ran late in the session on a progressively hotter device (cooldowns
could not restore nominal while plugged). Gen-length effect and thermal throttling
are **entangled** here — these are *hot-device* rates, not clean steady-state. The
clean short-generation decode reference is the prefill table above (all nominal, gen
64: 32–39 tok/s) and the cold/realistic runs (~35–39 tok/s). decode-2048 has n=4
(the 5th run was cut by the session-time overrun).

### Sustained stress (forced continuous decode, prompt 256, 5 min)
- **24 segments, 6144 tokens. tok/s 37.8 → 17.9 (−53%).** Steady ~37 tok/s for the
  first ~1300 tokens, then a knee down to ~18–19 tok/s by ~2300 tokens (~90 s in),
  holding there for the rest of the run.
- **`thermalState` reported `nominal` for all 24 segments** even as throughput
  halved → the coarse `ProcessInfo.thermalState` enum is a poor on-device throttle
  proxy; per-segment tok/s (decision 6) is the real signal.

\* **peak_mem caveat:** MLX `Memory.snapshot().peakMemory` is a process high-water
mark (monotonic within a session), so per-cell values reflect the running peak at
execution time, not the per-config peak. The meaningful figure is the **overall
session peak ≈ 2.2 GB** at the 2048-token configurations. (Reset-per-run is a noted
h3 improvement for the base-vs-LoRA comparison.)

## Conclusion

A reusable on-device performance baseline for the 4-bit base model now exists, with
error bars and scaling curves, built as a Mac-driven one-command rig reusable
verbatim for base-vs-Task-LoRA and on-device-training measurements.

Headline (steady-state, cool device): **~37 tok/s decode** at deployment context
sizes, **prefill 620–740 tok/s**, **app-launch-to-first-answer ≈ 1.7 s**, **peak
≈ 2.2 GB**. The prior n=1 anecdote (37.5 tok/s, peak 1.82 GB) sits squarely in this
distribution.

Two findings that shape the next pass:
1. **Sustained decode throttles hard on a plugged phone** (−53% after ~90 s), and
   `thermalState` does not report it. Any long-generation or training benchmark must
   read throughput directly, segment-by-segment, not trust the thermal enum.
2. **Clean steady-state long-decode curves are not obtainable while plugged** —
   the device cannot return to nominal between heavy cells. The pre-registered
   **unplugged-over-Wi-Fi follow-up** (plan decision 7) is the way to get a clean
   decode-length curve; deferred, not blocking.

For the base-vs-Task-LoRA comparison, the apples-to-apples channel is the **nominal
prefill table + realistic tier + cold start** (clean, reproducible); the stress
decay curve is the sustained-load stress test. The Task-LoRA, being rank-4 adapters
fused into the same 4-bit weights, is expected to land within noise of these base
numbers — which this baseline is now precise enough to test.
