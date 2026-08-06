# LongLaMP decoding fix and LL1–LL3 re-measurement

Branch A of the two options left open by
`experiments/2026-08-04-longlamp-degeneration-confound.md`. No retraining —
every adapter was already on disk; this round is eval-only.

## Hypothesis

LL1 and LL3 reported that Task-LoRA fine-tuning badly regresses on LongLaMP
(Review 0.343 → 0.182, Topic 0.281 → 0.131). The 2026-08-04 audit showed those
Task-LoRA arms generate ~4× gold length with 83/100 records past 600 words,
while the same base model with no adapter never exceeds 358 words. The
regression and a greedy-decoding repetition loop are perfectly confounded.

If the regressions are artifacts of that loop, then fixing decoding — applied
identically to every arm — should remove them. If they are real, the gap
survives the fix.

## Setup

**Step 0 — confirm the retraction in-container.** The 2026-08-04 numbers were
derived with a stubbed ROUGE scorer on the login host.

```
condor_submit condor/longlamp_degeneration_audit.sub      # cluster 179247, 3 CPU jobs
```

**Step 1 — new decoding flags.** `eval/eval_longlamp.py` previously exposed only
`--repetition-penalty` and called `generate()` with `do_sample=False`. Added
`--no-repeat-ngram-size`, `--do-sample`, `--top-p`, `--temperature`. Every
default is the transformers default and sampling-only parameters are omitted
from the `generate()` call under greedy, so all pre-existing result files stay
reproducible. Passing `--top-p`/`--temperature` without `--do-sample` is
refused rather than silently inert.

**Step 2 — dev sweep.** 7 configs × 3 tasks, `--limit 50`, `--split dev`.

```
python condor/gen_longlamp_decoding_sweep_subs.py
condor_submit condor/longlamp_decoding_sweep_smoke.sub    # cluster 179264, 2 jobs
condor_submit condor/longlamp_decoding_sweep.sub          # cluster 179265, 21 jobs
python eval/longlamp_length_check.py results/LongLaMP_*_dev_*_limit50.predictions.jsonl
```

Selection was **pre-registered**: admissible = degeneration rate at >600 words
near zero; among admissible, best dev ROUGE-1. Selection on dev only — LL1 was
right to refuse to sweep decoding against the test metric, and its error was
abandoning the sweep rather than moving it to dev.

**Step 3 — re-measure.** All three arms × three tasks, full test split,
identical decoding.

```
python condor/gen_longlamp_remeasure_subs.py
condor_submit condor/longlamp_remeasure_nrng3.sub         # cluster 179279, 9 jobs
```

## Result

### Step 0 — the retraction reproduces exactly

| Abstract | predicted (stub) | measured (container) |
|---|---|---|
| baseline degenerate | 30/100 | 30/100 |
| personalized degenerate | 21/100 | 21/100 |
| all users n=100 | +0.0267 | +0.026724 (t p=0.066, w p=0.0137) |
| **both-arms-clean n=63** | **+0.0013** | **+0.001316** (t p=0.843) |
| corr(length change, diff) | −0.950 | −0.9499 |

The pre-registered tripwire ("if the clean mean differs materially from +0.001,
re-examine the retraction") did not fire. **LL5's Abstract result stays
retracted.**

Step 0 also covered Review and Topic, which the retraction had not examined:

| task | Task-LoRA >600w | median words | gold median | both-arms-clean n |
|---|---|---|---|---|
| abstract | 30/100 | 142 | 144.5 | 63 |
| review | 83/100 | 817 | 211.5 | **9** |
| topic | 83/100 | 802 | 208.0 | **10** |
| base (no adapter) | 0/100 | 167 | 144.5 | — |

Abstract was the mild case. LL4's and LL6's nulls rest on clean subsets of n=9
and n=10.

### Step 2 — dev sweep

Greedy control reproduced the test-split degeneration rate closely (dev
90/88/28% vs test 83/83/30%), so dev is representative of the bug.

| config | worst-task >600w | mean dev R-1 (Task-LoRA arm) |
|---|---|---|
| greedy (control) | 90% | 0.2048 |
| **nrng3** | **6%** | **0.3670** ← selected |
| sample p0.95 t1.0 | 6% | 0.3628 |
| nrng4 | 8% | 0.3515 |
| nrng4 + rp1.1 | 10% | 0.3169 |
| sample p0.9 t0.7 | 36% | 0.3342 |
| nrng6 | 22% | 0.3197 |

`nrng3` follows mechanically from the pre-registered rule: tied-best
admissibility, best mean dev ROUGE-1. It also keeps decoding deterministic,
matching every other round in this project; the runner-up would have introduced
seed dependence for 0.004 less ROUGE.

### Step 3 — re-measured test table (n = full test split)

| task | arm | n | R-1 greedy | R-1 nrng3 | Δ | gen_tok greedy → nrng3 |
|---|---|---|---|---|---|---|
| Review (LL1) | floor | 1822 | 0.3282 | 0.3284 | +0.0002 | 378 → 381 |
| | baseline | 1822 | 0.3428 | 0.3427 | −0.0001 | 375 → 351 |
| | **Task-LoRA** | 1822 | 0.1818 | **0.3598** | **+0.1780** | 897 → 286 |
| Abstract (LL2) | floor | 4560 | 0.3973 | 0.3875 | −0.0097 | 138 → 140 |
| | baseline | 4560 | 0.4306 | 0.4108 | −0.0198 | 208 → 214 |
| | **Task-LoRA** | 4560 | 0.4131 | **0.4249** | **+0.0118** | 336 → 148 |
| Topic (LL3) | floor | 2452 | 0.2824 | 0.2904 | +0.0081 | 490 → 466 |
| | baseline | 2452 | 0.2808 | 0.2942 | +0.0133 | 579 → 522 |
| | **Task-LoRA** | 2452 | 0.1311 | **0.3114** | **+0.1803** | 957 → 294 |

**Headline — Task-LoRA minus BM25 baseline, both arms under identical decoding:**

| task | greedy | nrng3 |
|---|---|---|
| Review (LL1) | **−0.1610** | **+0.0171** |
| Abstract (LL2) | −0.0175 | +0.0141 |
| Topic (LL3) | **−0.1497** | **+0.0172** |

All three tasks flip from negative to positive.

**The control that makes this readable.** `nrng3` is close to a no-op on arms
that were not degenerating — Review's floor moved +0.0002 and its baseline
−0.0001 — and Abstract's baseline got *worse* (−0.0198). So the fix does not
lift all boats; it specifically rescues the arm that was running away. That
asymmetry in the effect is what licenses reading the flip as removal of an
artifact rather than as a decoding-induced inflation.

Degeneration rate on the re-measured test arms, >600 words:

| arm | Review | Abstract | Topic |
|---|---|---|---|
| floor | 0.1% | 0.0% | 1.5% |
| baseline | 0.1% | 0.1% | 5.7% |
| Task-LoRA | 5.3% | 0.0% | 3.8% |

Worst arm is 5.7%, down from 90%, and it is now the *baseline* rather than the
Task-LoRA. The asymmetry that produced the confound is gone.

## Conclusion

**LL1's and LL3's headline regressions were artifacts of greedy-decoding
repetition loops, not evidence that fine-tuning hurts on LongLaMP.** Under a
decoding config selected on dev and applied identically to every arm, the
Task-LoRA beats the BM25 baseline on all three tasks (+0.017 / +0.014 / +0.017).
That is a directional agreement with every LaMP round's Q1 result, where LL1–LL3
had stood as the lone counterexample.

The effects are small (~+0.015 R-1) and no significance testing was run on
them — these are point estimates on full test splits, in the same descriptive
convention as LL1–LL3 themselves. The honest summary is "fine-tuning no longer
regresses and is slightly positive," not "fine-tuning clearly helps."

### Limitations

1. **`nrng3` is an intervention, not a neutral fix.** Forbidding any trigram
   from repeating within a generation can block legitimate repeated phrasing in
   long prose. It won on dev ROUGE regardless, but the re-measured numbers
   should be described as "under constrained decoding."
2. **Residual degeneration is not zero** (worst arm 5.7%). Long-form means on
   these tasks should still be read alongside `eval/longlamp_length_check.py`.
3. **The decoding config was selected on the `_temporal` dev split and applied
   to `_user` test.** Same adapters, same base model, different partition. No
   dev record enters the test measurement either way, but the transfer is an
   assumption.
4. **LL4–LL6 are NOT re-measured by this round.** The retraction of LL5 stands
   on its own evidence and is unaffected; but whether a *clean* per-user effect
   exists on LongLaMP is now an open question that this round does not answer.
5. Single seed; ROUGE-1 only as the headline (ROUGE-L recorded in the result
   JSONs).

### What this does not change

The LL5 Abstract retraction. That result was a within-round baseline-vs-
personalized asymmetry, and Step 0 confirmed it in-container to five decimal
places. Fixing decoding removes the *mechanism* that produced it, which is
exactly why LL4–LL6 would need re-running to say anything new about per-user
personalization on LongLaMP.

## Next

- **LL4–LL6 re-measurement under `nrng3`** — ~303 jobs (1 baseline + 100
  per-user × 3 tasks), no retraining. This is the round that would establish
  whether a real per-user effect exists on long-form output once the lottery is
  removed. Both arms already degenerate at comparable rates post-fix, so the
  confound that killed LL5 would not recur in the same shape.
- **BFCL is unaffected** — those numbers measure the adapter, not decoding.
- Paper: LL1–LL3's sections need the before/after table, and the regression
  narrative needs rewriting rather than patching.

## Provenance

- Clusters: 179247 (audit, 3 jobs), 179264 (smoke, 2), 179265 (dev sweep, 21),
  179279 (re-measure, 9). **35 jobs, 35 exit 0, zero infra failures** — tyr1/modi
  exclusions and the Blackwell guard were baked into every generated sub from
  the first submit.
- Code: `eval/eval_longlamp.py` (new flags), `eval/longlamp_length_check.py`
  (new), `condor/gen_longlamp_decoding_sweep_subs.py`,
  `condor/gen_longlamp_remeasure_subs.py`. Commits `91b4e77`, `f498602`.
- Original plain-greedy LL1–LL3 result files are untouched on disk; the new
  results live at `..._seed0_nrng3.json` alongside them.
