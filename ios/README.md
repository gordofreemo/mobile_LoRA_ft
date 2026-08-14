# iOS app and vendored MLX packages

Everything that runs on the phone lives here. The app is a modified copy of Apple's `LLMEval`
example with our benchmark harness bolted on. It measures inference, trains real LoRA adapters on
device, and writes one JSON line per measurement into its own `Documents/` container.

## What is vendored

| Path | What it is |
| --- | --- |
| `mlx-swift-examples/` | [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) pulled in with `git subtree`, upstream base `378f244`. Holds the app and our harness. |
| `mlx-swift-lm-local/` | Local SPM override of [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm). Model and training code, with three of our edits. |
| `mlx-swift/` | Local SPM override of [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) carrying the NAX kernel patch. See `VENDORED.md` and `LOCAL_PATCHES.md`. |
| `BGProbe/` | Standalone control app for the background scheduling round. Registers a `BGProcessingTask` and logs a heartbeat, nothing else. |

Vendoring rather than patching at build time means a plain `git clone` reproduces the harness with no
apply step, and our edits keep normal file history.

Our edits to `mlx-swift-lm-local`: gradient checkpointing in `Models/SmolLM3.swift`
(`checkpointGroupSize`), a seedable batch iterator in `LoraTrain.swift` (`LoRATrain.shuffleSeed`, so
paired A/B arms see identical batch order), and an early release of the weights dictionary in
`Load.swift`.

Our edits to `mlx-swift-examples`: the five files under `Applications/LLMEval/Benchmark/`, the model
configuration and telemetry hooks in `ViewModels/LLMEvaluator.swift`, and
`LLMEval-Info-Additions.plist`, which carries the array valued Info.plist keys that Xcode's
`INFOPLIST_KEY_*` synthesis cannot express.

The kernel fix in `mlx-swift/` is upstream now, merged as
[ml-explore/mlx#4051](https://github.com/ml-explore/mlx/pull/4051). The vendored copy predates the
release that contains it, so it stays until the pin moves. It is inert by default: `enable_nax_n()`
returns 0 unless a run asks for it with `--nax-arm on`.

## Build, install, run

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples

# -skipMacroValidation is required, the build fails on MLXHuggingFaceMacros without it
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination "id=$UDID" -derivedDataPath ./build \
  -allowProvisioningUpdates -skipMacroValidation DEVELOPMENT_TEAM=$TEAM build

xcrun devicectl device install app --device $UDID build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device $UDID $BUNDLE --benchmark
```

On first generation the app downloads its model from Hugging Face over Wi-Fi, about 1.7 GB for the
4-bit 3B. It survives reinstalls.

Pull telemetry with `devicectl device copy from`, then aggregate on the Mac:

```bash
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer \
  --domain-identifier $BUNDLE --source Documents/bench_metrics.jsonl \
  --destination ../../results/ondevice/bench_metrics_$(date +%F).jsonl
python ../../eval/bench_aggregate.py ../../results/ondevice/bench_metrics_*.jsonl
```

Each benchmark mode writes its own JSONL so a run in progress never touches another round's file.
The mode is chosen by launch argument, listed in the repository README and in `CLAUDE.md`.

## Things that will cost you an evening

* Install over the existing app. Never `devicectl device uninstall`: it wipes the cached model and
  any side loaded user data, and it resets the developer trust, which then needs a manual tap in
  Settings before the app will launch again.
* Kill any resident process before launching with new arguments. A running instance silently ignores
  them and the launch looks like it worked.
* `devicectl device process launch --console` forwards SIGTERM to the app, so killing the monitoring
  session kills the run. Launch detached and poll the on-device JSONL instead.
* Keep the vendored directory named exactly `mlx-swift`. SPM matches local overrides by directory
  basename against the remote URL tail, and a different name silently pulls unpatched MLX.
* The NAX kernels are compiled at runtime from `Source/Cmlx/mlx-generated/quantized_nax.cpp`. Editing
  the Metal header in `mlx/backend/metal/kernels/` changes nothing and the results reproduce to the
  last digit, which reads exactly like a real negative result.

## Updating from upstream

```bash
git subtree pull --prefix=ios/mlx-swift-examples \
  https://github.com/ml-explore/mlx-swift-examples <tag-or-sha> --squash
```

The two SPM overrides are plain directories, so bumping them means copying a fresh checkout over the
old one, minus `.build` and `.git`, and reapplying the edits listed above.
