# Fine-Tuning a 3B-Parameter LLM on a Smartphone: Characterizing Sustained Training

Andrew Geyko, Marius Mosbach, André Brinkmann

This repository contains the code and measurement data for the paper.

We fine-tune SmolLM3-3B with LoRA on an iPhone 17 Pro and measure complete training runs:
memory, the time breakdown of a training step, thermal throttling over hours of training, and
energy. The backward pass over the frozen 4-bit weights takes most of each step. MLX had a
matrix-unit kernel for it that was never dispatched and was incorrect, and our repair was merged
upstream as [ml-explore/mlx#4051](https://github.com/ml-explore/mlx/pull/4051). The per-user
adapters are evaluated on LaMP with the protocol of OPPU, trained both on a GPU cluster and on the
phone.

## Repository layout

| Directory | Contents |
|---|---|
| `ios/` | the iPhone benchmark app and the MLX packages it builds against (see `ios/README.md`) |
| `scripts/` | drivers for the device experiments |
| `train/` | cluster training with Transformers and PEFT, dataset builders |
| `eval/` | evaluation, aggregation of the device telemetry, and the plotting scripts |
| `oppu_replication/` | wrapper around the OPPU release used for the per-user evaluation |
| `condor/` | HTCondor submit files and the generators that write them |
| `data/` | dataset download and per-user corpus builders |
| `results/` | measurements and predictions, with the phone's telemetry in `results/ondevice/` |
| `figures/` | figures generated from `results/` |
| `experiments/` | notes on individual experiment rounds |

## Requirements

The device experiments need a Mac with Xcode (we used Xcode 26.5), an iPhone 17 Pro with Developer
Mode enabled, and an Apple developer team for code signing. All reported device measurements were
taken on iOS 26.6.2.

The cluster experiments run in the Docker image built from `Dockerfile` and `requirements.txt`,
published as `ghcr.io/gordofreemo/smollm3-train:ver4`, on an HTCondor pool with GPUs.

## Data

```bash
condor_submit condor/download_model.sub          # SmolLM3-3B into data/models/
python data/download_lamp.py --split-type user   # task adapter training data
python data/download_lamp.py --split-type time   # per-user histories
condor_submit condor/build_dataset.sub           # BM25 retrieval into JSONL
```

The per-user evaluation uses the data and code released by OPPU, placed under
`data/oppu_release/` and `third_party/OPPU/`. Neither LaMP nor the OPPU release is redistributed
here. LaMP-6 needs the licensed Avocado corpus and is not supported.

The fused 4-bit model trained on the phone is on the Hugging Face Hub as
[`ageyko/SmolLM3-3B-a1lamp-4bit`](https://huggingface.co/ageyko/SmolLM3-3B-a1lamp-4bit).

## Device experiments

Build and install the app as described in `ios/README.md`, then launch it with a benchmark
argument. Each run appends one JSON line per measurement to a file in the app's `Documents/`
container:

```bash
xcrun devicectl device process launch --device $UDID $BUNDLE --benchmark-train-perop
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer \
  --domain-identifier $BUNDLE --source Documents/train_bench_metrics_perop.jsonl \
  --destination results/ondevice/train_bench_metrics_perop_$(date +%F).jsonl
```

`--nax-arm on|off` selects the backward kernel in any benchmark mode. The scripts in `scripts/` run
whole experiments unattended. They set our device and bundle identifiers (`DEV`, `BID`) at the top,
which you need to replace with your own.

| Paper section | Experiment | Analysis |
|---|---|---|
| 4, memory | `scripts/run_h14_capsweep.sh`, `--benchmark-train-granularity` | `eval/h14_capsweep_summary.py`, `eval/plot_granularity_split.py` |
| 5, step breakdown | `--benchmark-train-perop`, `--benchmark-train-perop-capture` for GPU captures | `eval/perop_aggregate.py`, `eval/plot_perop.py` |
| 5, kernel repair | `--benchmark-nax-ab`, `scripts/run_h15_b_kernelab.sh` | `eval/naxab_aggregate.py`, `eval/h16dq_summary.py` |
| 5, runtime audit | | `eval/runtime_quant_backward_audit.py` |
| 6, complete runs | `scripts/run_h15_a_cost.sh`, `scripts/run_h14_repeats.sh` | `eval/e2e_aggregate.py`, `eval/plot_hook_run.py` |
| 6, adapter configuration | `scripts/run_h14_rank_sweep.sh`, `scripts/run_h15_d_depth.sh` | |
| 6, sustained training | `--benchmark-thermal-cooldown`, `scripts/run_h15_e_schedule.sh` | `eval/thermal_aggregate.py`, `eval/plot_sustained_split.py` |
| 6, energy | `scripts/run_h15_c_energy.sh`, `scripts/run_h15_f_rerun.sh` | `eval/e2e_aggregate.py` |
| 7, adapters trained on the phone | `scripts/h13/` | `eval/h13_score.py` |

Energy runs start unplugged at the battery level given in each script header.

## Cluster experiments

```bash
condor_submit condor/train_1ep.sub          # task adapter
condor_submit condor/eval_lamp.sub          # LaMP evaluation with BM25 retrieval
condor_submit condor/eval_lamp_floor.sub    # without retrieval
python condor/gen_oppu_subs.py              # per-user adapters under the OPPU protocol
python eval/summary.py                      # collect results/*.json into a table
```

Per-user experiments train one adapter per user, so their submit files are generated. Edit the
generator rather than the `.sub` files. `eval/q4b_table8_compare.py` compares the per-user results
at 4-bit precision.

Each script prints its provenance on the first line and refuses to overwrite an existing output
without `--overwrite`. Evaluation uses greedy decoding with seed 0.

## Results

`results/` holds one flat JSON file per run together with per-example predictions. Each record
carries the git commit, library versions and host. The device telemetry in `results/ondevice/` is
the raw data behind the figures, and
`results/ondevice_e2e_smollm3_a1lamp_nax-on_36L_2026-09-20.json` summarizes the complete training
runs.

## Citation

```bibtex
@misc{geyko2026finetuning,
  title  = {Fine-Tuning a 3B-Parameter {LLM} on a Smartphone: Characterizing Sustained Training},
  author = {Geyko, Andrew and Mosbach, Marius and Brinkmann, Andr{\'e}},
  year   = {2026},
}
```

## License

The code is released under the MIT license (see `LICENSE`). The measurement data in `results/` and
the figures in `figures/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The packages vendored under `ios/` keep
their upstream MIT licenses.
