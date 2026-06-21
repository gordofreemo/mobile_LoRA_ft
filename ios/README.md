# iOS — on-device benchmark app (vendored)

`ios/mlx-swift-examples/` is the [`ml-explore/mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples)
repo **vendored into this repo via `git subtree`** (upstream base **`378f244`**).
It carries our on-device base-inference benchmark harness for `SmolLM3-3B-4bit`
(design: `experiments/2026-06-21-ondevice-base-inference-plan.md`; results:
`experiments/2026-06-21-ondevice-base-inference.md`).

It was a gitignored loose clone until 2026-06-21; converting it to a subtree makes
our edits ordinary tracked files with full history and makes a single `git clone` of
this repo reproduce the harness — no patch/apply ritual. **The ML libraries are *not*
vendored here**: they come from the pinned `ml-explore/mlx-swift-lm` SPM package
(`Package.resolved`).

## What's ours (the rest is upstream)

- `Applications/LLMEval/Benchmark/BenchmarkSupport.swift` — flat-JSON schema,
  `DeviceState` snapshot, grid constants, `measureGeneration` (drives `TokenIterator`
  directly so forced fixed-length = ignore EOS to `maxTokens`).
- `Applications/LLMEval/Benchmark/LLMEvaluator+Benchmark.swift` — orchestration:
  cold-start, prefill/decode curves, realistic LaMP-3 tier, sustained stress, cooldown,
  launch-arg detection, record writer. `app_build` tag: **`smollm3-ondevice-bench-h2`**.
- `Applications/LLMEval/ViewModels/LLMEvaluator.swift` — `modelConfiguration` set to
  `mlx-community/SmolLM3-3B-4bit`; extracted `appendBenchLine` / `benchLogLine`.
- `Applications/LLMEval/Views/ContentView.swift` — `.task` auto-runs the benchmark on
  the `--benchmark*` launch args (skips the UI preload).
- `mlx-swift-examples.xcodeproj/project.pbxproj` — adds the two `Benchmark/` files to
  the LLMEval target's `membershipExceptions`.

## Workflow

**Edit the harness → ordinary `git commit`.** No patch regeneration. Bump
`BenchConstants.appBuild` whenever harness logic changes (it's stamped into every
telemetry record for provenance).

**Pull upstream updates** (rare — we pin a known-good base):

```
git subtree pull --prefix=ios/mlx-swift-examples \
  https://github.com/ml-explore/mlx-swift-examples <tag-or-sha> --squash
```

Resolve any conflicts in our five files above, commit, rebuild.

> `git subtree` ships as a git-core script (`/Library/Developer/CommandLineTools/usr/libexec/git-core/git-subtree`);
> `git subtree --help` may lack a manpage but the subcommands work.

## Build / install / run

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
# -skipMacroValidation is required (MLXHuggingFaceMacros), -derivedDataPath ./build
# keeps build output inside the (gitignored) build/ dir:
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination 'id=00008150-000674C60A3B401C' \
  -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=JGW9U9Y36Y build
xcrun devicectl device install app --device 00008150-000674C60A3B401C \
  build/Build/Products/Debug-iphoneos/LLMEval.app

# Full grid (cold + prefill + decode); phone unlocked, one Face-ID at start:
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark
# Tail (realistic LaMP-3 tier + 5-min stress) — separate launch so the long cells
# don't overrun a single --console session:
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark-tail
# Cold-start variance (run 3× from separate launches):
xcrun devicectl device process launch --console --terminate-existing \
  --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y --benchmark-cold
```

> Reinstalling a rebuilt app on a free personal team can rotate the provisioning
> profile, requiring a one-time on-device re-trust (Settings → General → VPN & Device
> Management → Trust the developer app) before `launch` succeeds.

Pull telemetry + aggregate:

```
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl \
  --destination results/ondevice/bench_metrics_smollm3-4bit-base_<date>.jsonl
python eval/bench_aggregate.py results/ondevice/bench_metrics_*.jsonl \
  --out results/ondevice_base_smollm3_4bit_<date>.json
```
