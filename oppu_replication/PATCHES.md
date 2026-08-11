# Deviation ledger vs third_party/OPPU @ 87f8c69

`run_task_lora.py` / `run_oppu.py` are patched copies of `task_LoRA.py` /
`OPPU.py`. Training and eval logic (recipe values, prompt construction, BM25
usage, loss masking, sampled decoding params) is theirs, line for line.
Regenerate the diff anytime: `diff -u third_party/OPPU/task_LoRA.py
oppu_replication/run_task_lora.py`.

| # | Patch | Why |
|---|---|---|
| P1 | dropped `import bitsandbytes as bnb` | unused upstream; not in our container |
| P2 | `sys.path` inserts for vendored `rank_bm25.py` (0.2.2, Apache-2.0, `rank_bm25.LICENSE`) and their `utils.py` | container lacks rank_bm25; utils stays upstream, imported in place |
| P3 | `extrat_product_review` → `extract_product_review`, `extrat_tweet_paraphrasing` → `extract_tweet_paraphrasing` | upstream NameError — product_rating and tweet_paraphrase are unrunnable as released |
| P4 | model default → local `data/models/SmolLM3-3B`; the Llama-2 token surgery (`eos="</s>"`, `pad='[PAD]'`) only runs when `</s>` exists in the vocab, else `pad = eos` | SmolLM3 has no `</s>`; intent preserved (left-pad with eos, stop on eos) |
| P5 | test-user file per task; tweet_paraphrase reads `user_more_100_history.json` | their release ships that name; their code's hardcoded name crashes |
| P6 | `save_steps=int(1e9)` | newer transformers rejects float >1 |
| P7 | removed `print(train_data)` | dumps the full training corpus to stdout |
| P8 | output layout: ckpts under `train/checkpoints/oppu_rep/<task>/`, preds under `results/oppu_rep/<task>/`; refuse-to-overwrite; `_limit*`/`--tag` smoke suffixes | repo conventions; upstream reused a shared `outputs/` dir and wrote into `./ckpt` `./output` |
| P9 | provenance banner + meta sidecar JSONs; post-training `lora_B` nonzero guard (exit 2), prediction-count guard (exit 3) | repo conventions; exit-0-with-garbage protection |
| P10 | per-query `{id, pred, gold}` JSONL alongside their `{task, golds, model}` JSON; inner output loop uses `j` (upstream shadowed the user index `i`) | feeds both stats layers; no behavior change |
| P11 | `transformers.set_seed(--seed)` immediately before each generation phase | their sampled decoding (do_sample, top_k=10, T=0.1, top_p=0.9) has no seed → irreproducible; decoding params untouched |
| P12 | `run_oppu.py`: `--user-start/--user-end` sharding for cluster parallelism; per-user diagnostics (fresh-adapter zero check + merged-base weight-hash canary); adapter `unload()` after each user | upstream loops all users in one process, calling `get_peft_model()` repeatedly on the same model object — sharding + unload guarantee per-user independence; the canary logs if upstream's pattern would have differed |
| P13 | `run_task_lora.py`: `--limit` (test users) / `--limit-train` (train users) | smoke runs, `_limitN` collision-free naming per repo convention |
| P14 | `str()` coercion at the `get_first_k_tokens` truncation sites | citation/scholarly release data carries int `date` fields; their `.split()` helper crashes on them |
| P15 | TrainingArguments kwargs filtered against the installed signature; dropped keys logged loudly | the container's transformers (5.9.0) removed `group_by_length` (all 7 first-smoke jobs died on it); any key it drops is a printed, recorded deviation — confirmed drop list: `['group_by_length']` (their length-grouped batching, an efficiency/batch-composition feature, not a recipe value) |
| P16 | product_rating + scholarly_title train at per-device 4 × grad-accum 4 (effective 16); other tasks keep their per-device 16 | batch 16 × 2048 tokens OOMs 40 GB GPUs on the two long-document tasks (smoke cluster 180670); their paper's Table 5 says batch is "3-16, task-dependent" while the code hardcodes 16 — effective batch preserved |

Known upstream behaviors kept deliberately (fidelity): dead `"out_proj"`
target-module name (matches nothing on Llama-2 or SmolLM3 → effective q/v/k);
`prepare_model_for_kbit_training` on a non-quantized bf16 model (casts params
to fp32); in-place profile truncation (768 tokens at train, 368 at eval);
`max_new_tokens=200` even for 1-token rating answers; norm modules cast to
fp32 after Trainer init.
