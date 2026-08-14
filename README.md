# On-device LoRA training for SmolLM3-3B

Research code for two connected questions. First, how far LoRA fine-tuning gets a 3B model on
personalization benchmarks (LaMP and LongLaMP). Second, what it costs to train those adapters on a
phone rather than a cluster, and what has to change to make that practical.

The base model is [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and it stays frozen
throughout. Cluster training uses Transformers and PEFT under HTCondor. The device work runs
[MLX](https://github.com/ml-explore/mlx-swift) on an iPhone 17 Pro.

Status: unpublished, paper in preparation. Numbers here are current as of August 2026 and can still
change. Per round write-ups are in `experiments/`, raw results in `results/`.

## Findings so far

* Task adapters help. A LoRA trained on the LaMP corpus, with BM25 retrieval in the system prompt,
  gains 0.11 accuracy on LaMP-3 and 0.07 and 0.13 ROUGE-1 on LaMP-4 and LaMP-7 over the same prompt
  without it. A second adapter covering all seven LaMP tasks scores above Llama-3.1-70B-Instruct with
  retrieval on every one of them.
* Per-user adapters were null in our own evaluation, about twenty rounds of them. Running OPPU's
  released protocol unmodified on the same model gives a clear positive on movie tagging (accuracy
  0.4933 to 0.5845). Subsampling that effect down to our evaluation shape, one query per user over
  100 users, detects it 24% of the time. The nulls were an evaluation power problem, not a model or
  recipe problem.
* Training a real adapter on the phone works and is bounded by energy. A 550 example user adapter
  finishes on one battery charge. A 987 example one does not: it dies at 49% of its iterations having
  spent 90% of the battery. The kernel fix below cuts the cost by about a third but does not remove
  the ceiling.
* Almost none of that cost is the adapter. The backward pass is 78% of a training iteration and is
  90% quantized matmul against the frozen base weights, while the LoRA gemms plus the entire
  optimizer step come to 4.2%. PEFT buys memory and storage on device, not compute.
* Scheduling around the thermal limit does not pay. Ten minute bursts with two minute gaps reach
  0.755x the throughput of training continuously, and pacing every iteration reaches 0.877x. Training
  already sits at the sustained power envelope, so there is no headroom to reclaim. Background
  execution is not an option either: a `BGProcessingTask` grants roughly 2.3 seconds of GPU access
  before revoking it.
* Most of that cost sat behind one gate. MLX enabled its neural accelerator path for quantized
  matmul only when the weights were transposed, which is every case except the backward pass, so the
  backward matmul ran on a generic kernel worth roughly 60% of an iteration. The kernel behind the
  gate had two bugs of its own, since nothing had ever reached it. With both fixed, a complete user
  adapter trains 1.93x faster with the same loss curve. Merged upstream as
  [ml-explore/mlx#4051](https://github.com/ml-explore/mlx/pull/4051).

## Setup

Cluster jobs run in `ghcr.io/gordofreemo/smollm3-train:ver4`, built from `Dockerfile` and
`requirements.txt`. Bump the tag in every `condor/*.sub` when either changes.

```bash
condor_submit condor/download_model.sub     # SmolLM3-3B into data/models/
python data/download_lamp.py --split-type user   # Task-LoRA training data
python data/download_lamp.py --split-type time   # per-user data, same users across splits
condor_submit condor/build_dataset.sub      # LaMP -> BM25 retrieved JSONL
```

LaMP-6 and LongLaMP's email task need the licensed Avocado corpus and are not supported. The OPPU
replication expects their released data under `data/oppu_release/` and their code under
`third_party/OPPU/`, neither of which is redistributed here. Their release carries no license, so it
is used locally only.

Device work needs a Mac with Xcode and an iPhone with Developer Mode on. See `ios/README.md`.

## Cluster experiments

```bash
condor_submit condor/train_1ep.sub          # Task-LoRA -> train/checkpoints/a1_lamp_1ep_seed0/
condor_submit condor/eval_lamp.sub          # LaMP-{3,4,7}, BM25 k=4
condor_submit condor/eval_lamp_floor.sub    # no-profile floor
condor_submit condor/eval_bfcl.sub          # BFCL tool calling, capability check
python eval/summary.py                      # collect results/*.json into a table
```

Per-user rounds run one adapter per user, so their submit files are generated rather than written by
hand. Edit the generator, not the `.sub` files:

```bash
python condor/gen_newtask_subs.py --all     # per-user rounds on the five later LaMP tasks
python condor/gen_warm_subs.py --all        # warm start arms, all seven tasks
python condor/gen_oppu_subs.py              # OPPU replication
```

Results from a per-user round are consolidated with `eval/aggregate_user_predictions_newtask.py` and
compared with `eval/paired_compare_per_user.py`, which reports the paired difference, win/tie/loss
and conventional statistics per user.

Every script prints its provenance on the first line and refuses to overwrite an existing output
without `--overwrite`. Smoke runs take `--limit N` and write to `_limitN` filenames so they cannot
collide with a full run. Evaluation is greedy at seed 0.

## Device experiments

Build, install and launch with a benchmark argument. The app writes one JSON line per measurement
into its `Documents/` container.

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination "id=$UDID" -derivedDataPath ./build \
  -allowProvisioningUpdates -skipMacroValidation DEVELOPMENT_TEAM=$TEAM build

xcrun devicectl device install app --device $UDID build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device $UDID $BUNDLE --benchmark-train-perop
```

The main benchmark arguments are `--benchmark` and `--benchmark-stress-capped` for inference,
`--benchmark-train-e2e --user <id> --condition <C0|C2>` for a full user adapter,
`--benchmark-train-tokentime` for the cost model,
`--benchmark-train-granularity --granularity-k K` for checkpoint granularity,
`--benchmark-thermal-cooldown` for the thermal arms, `--benchmark-train-perop` for the per phase
breakdown, and `--benchmark-nax-ab` for the kernel A/B. `--nax-arm on|off` selects the kernel in any
of them and routes telemetry to a separate file.

Pull and aggregate:

```bash
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer \
  --domain-identifier $BUNDLE --source Documents/train_bench_metrics_perop.jsonl \
  --destination results/ondevice/train_bench_metrics_perop_$(date +%F).jsonl
python eval/perop_aggregate.py results/ondevice/train_bench_metrics_perop_*.jsonl
```

Full device setup, including the vendored and patched MLX packages, is in `ios/README.md`.

## Layout

```
train/       training entry points and per run JSON configs
eval/        evaluation harnesses, paired statistics, device aggregators and plots
data/        dataset download, user pool selection, per-user corpus builders
condor/      HTCondor submit files and the generators that emit them
oppu_replication/   wrapper around the OPPU release, with a patch ledger
ios/         the iPhone app and the vendored MLX packages it builds against
scripts/     overnight sequencing scripts for device runs
results/     one flat JSON per run plus per-example predictions
experiments/ one markdown file per round: hypothesis, setup, result, conclusion
```

Result files are single level JSON so a directory of them loads straight into a dataframe, and each
one carries `git_commit`, library versions, job ids and hostname. Predictions sit next to them as
JSONL. Most `experiments/*.md` files are not committed, since they carry unpublished analysis.

## Artifacts

* Fused 4-bit device model: [`ageyko/SmolLM3-3B-a1lamp-4bit`](https://huggingface.co/ageyko/SmolLM3-3B-a1lamp-4bit)
* Kernel fix upstreamed to MLX: [ml-explore/mlx#4051](https://github.com/ml-explore/mlx/pull/4051)

## References

* LaMP benchmark: Salemi et al., [lamp-benchmark.github.io](https://lamp-benchmark.github.io)
* LongLaMP: [arXiv:2407.11016](https://arxiv.org/abs/2407.11016)
* OPPU, the per-user recipe replicated here: Tan et al., [arXiv:2402.04401](https://arxiv.org/abs/2402.04401)
* BFCL: [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)
* MELT, the model for the device measurements: Laskaridis et al., MobiCom 2024,
  [arXiv:2403.12844](https://arxiv.org/abs/2403.12844)
* Apple on-device fine-tuning: [arXiv:2510.03425](https://arxiv.org/abs/2510.03425)
