# h13 — on-device validation of the OPPU personalization effect (movie tagging)

Pinned 2026-08-19 via /grill_me. Goes in the paper (self-contained section, deletable
without unstitching the two acts). Claude implements everything. No gates (pre-reg retired);
descriptive stats, user decides meaning.

## Claim

The one personalization effect this project has measured with adequate power — OPPU faithful
replication, movie tagging (LaMP-2M), +0.0912 query-level / +0.0336 grouped — survives
end-to-end **on-device** training, evaluated under **4-bit deployment conditions on the phone**.

## Design decisions (settled in the grill, do not relitigate)

1. **Fresh device runs, not the saved h5/E2E adapters.** The saved LaMP-3 adapters are the wrong
   artifacts: our LaMP-3 eval shape detects even the largest effect ever produced only 24% of the
   time. Personalization is measured the OPPU way or not at all.
2. **Arena = OPPU replication protocol**: their splits, their prompts, their evaluator (unchanged
   scoring layer), movie tagging only. Device arm trains with the **R5 recipe bundle** — the P19
   ablation proved it still yields +0.0763 inside their protocol, so no hot-recipe port is needed.
3. **Users = all previously cluster-trained movie users, as an anytime queue.** Order frozen here
   before the first run: descending **predicted paired-queries-per-device-hour** (cost from
   per-user profile token counts via the NAX-ON law `wall_s ≈ 0.0101·tokens + 222`, plus eval
   time), tiebreak by query count. Any prefix of the queue is a complete, reportable result.
   Caveat to carry into the writeup: the early prefix is biased toward small-profile users.
4. **Device recipe = max-faithful.** 36 layers (not the h5-era 28), q+v r=8 α=16, AdamW 1e-5
   wd 0.01, 3 epochs, GC on, **effective batch 8 via gradient accumulation** (accumulation port
   from the h12 path; memory must NOT increase — verify peak_mem in smoke, fall back to batch 1
   and report if it does), **cosine + 3% warmup** via per-step learningRate assignment.
   Remaining deviations are exactly the forced ones: 4-bit base, no dropout.
5. **Mac control on EVERY user.** Same MLX code path, same 4-bit base, same recipe, same
   `LoRATrain.shuffleSeed` batch order. Runs on the M3 in parallel with device nights.
6. **Eval plane = the phone, 4-bit MLX, all arms.** Four arms per user — RAG baseline,
   cluster-trained adapter (PEFT→MLX converted), Mac-control adapter, device adapter — all
   generate ON-DEVICE with their prompts and their decoding config, sampled seed 0 (seeded in
   MLX). One stack ⇒ per-query pairing is clean. Predictions pulled and scored off-device with
   their unchanged evaluator. **The published bf16/HF numbers (+0.0912, +0.0763) do NOT carry
   over** — the RAG-vs-cluster gap is re-measured on this plane; never mix the two planes.
   Side product: per-query inference telemetry with an unfused per-user LoRA stacked (revives the
   deferred adapter-inference item).

## Phases

- **Phase 0 — smoke.** Smallest queue user, end-to-end: data builder → side-load → ~20-step
  device train → adapter save → on-device 4-arm eval → pull → convert → score. Every pipe joint
  once before any paid night.
- **Phase 1 — eval-plane re-establishment (no training).** Convert cluster adapters for the first
  ~5 queue users; run RAG + cluster arms on-device; score. Answers "does the effect survive 4-bit
  MLX deployment at all" for one evening of inference. If the effect dies here, stop — that is
  the finding.
- **Phase 2 — anytime queue.** Per user: device train → ~5 min cooldown (h10 t95 ≈ 346 s) →
  on-device eval of remaining arms → pull everything. Mac control trains in parallel and is ready
  by eval time. A user is not "done" until trained + evaluated + pulled.

## Reporting kit (complete; nothing pooled)

- Effect table per prefix: per-query paired diffs for cluster−RAG, device−RAG, mac−RAG,
  device−cluster, device−mac; means, grouped per-user stats, W/T/L, conventional p-values.
- Agreement / manipulation check: prediction-change rate between every arm pair (byte + scored),
  so parity cannot hide as "both arms barely move".
- Training fidelity: device-vs-Mac loss curves (same seed/batch order), final-loss deltas.
- Systems telemetry: E2E-style training records (a free movie-task extension of the C0 cost-law
  family) + per-query inference telemetry.

## Verify on conduit BEFORE building (flagged during the grill)

1. What the cluster movie OPPU+RAG arm actually stacks — if their protocol includes a task
   adapter under the user adapter, the device/Mac base must be a 4-bit MLX fusion of THEIR task
   adapter (same pipeline that made `ageyko/SmolLM3-3B-a1lamp-4bit`), not a1lamp, not bare base.
2. Their trainer's loss masking + exact per-user training text, so the Mac-side data builder
   emits byte-identical sequences to what the P19 R5-ablation arm consumed.
3. Which movie users have existing R5-ablation adapters (P19 was movie-only; confirm coverage).
4. Per-user test-query counts + profile token counts → compute and freeze the queue order.
5. What `oppu_rep_score --eval-only` (P20) actually does; where externally-generated predictions
   enter its scoring layer.

## Build list

Device harness (new mode, new `app_build`, new JSONLs; `--nax-arm on`):
36-layer constant in the new mode, accumulation port, cosine+warmup LR, movie-data loader
(side-load format), **on-device eval mode** (prompts JSONL → seeded sampled generation →
predictions JSONL + telemetry; loads user adapter unfused over the 4-bit base).
Mac side: their-protocol data builder, PEFT→MLX reverse converter (mapping documented in
`convert_mlx_adapter_to_peft.py`; verify by round-trip), Mac-control training config
(`train_task_mlx.py` lineage), scoring plumbing.

## Standing traps that apply

Wall charger, never Mac USB, for plugged runs. Kill resident PID before every launch. Sequencer
under nohup + caffeinate, detached. Auto-Lock Never. Never uninstall. Side-load via
`devicectl device copy to`. Seeded batch iterator for device/Mac loss comparability. Scoring off
the login node.
