# Research Project — On-Device LLM Personalization via PEFT

Fine-tuning **SmolLM3-3B** for personalized instruction following via a two-stage
LoRA pipeline: a **Task-LoRA** (LaMP-{3,4,7}, BM25 profile in `system`) plus a
per-user **User-LoRA** on time-ordered history, stacked at inference. Phases 1 & 2
ran on the cluster; Phase 3 deploys to a real iPhone.

## Active work (as of 2026-07-13)

- **Phase 3 — background-scheduled on-device training — h6 HARNESS BUILT + VALIDATED 2026-07-13, REAL 7-DAY RUN LIVE (restarted again, now on schema v5) as of 2026-07-14T16:09:08Z, cap 2026-07-21T16:09:08Z.** Design doc: `experiments/2026-07-13-ondevice-bg-training-plan.md`, pinned 2026-07-13 via `/grill_me`. Follow-on to the E2E plan below, NOT a replacement — the E2E plan's foreground/screen-on 14-run matrix still runs as designed and stays primary. This round asks a deployability question instead of a cost question: does per-user training survive real Apple `BGProcessingTask` OS scheduling (no `isIdleTimerDisabled` foreground hack)? **Two co-equal headlines:** (1) calendar-time-to-complete vs device-compute-time-to-complete for one full adapter trained under real (non-forced) scheduling; (2) wake-scheduling characterization (wake count, gap distribution, per-wake time budget, charging correlation). **Subject:** single user S=`u00008075`/405 (smallest sample user, to maximize odds of finishing within the cap). **`requiresExternalPower=true`** (realistic "charging overnight" scenario; property is `requiresExternalPower` on `BGProcessingTaskRequest`, NOT `requiresExternalPowerConnected` as the design doc assumed — corrected against the actual SDK header during implementation). **Checkpoint/resume covers LoRA weights + iteration counter ONLY, persisted every 10 iterations** via chunked `LoRATrain.train()` calls (single blocking call — can't checkpoint mid-call). **Adam's optimizer moments reset to zero every wake** — `MLXOptimizers.AdamW`'s internal m/v state is `internal`-access with no public getter/setter (verified against `mlx-swift`'s `Source/MLXOptimizers/Optimizers.swift`); hand-rolling a replacement optimizer to work around this was considered and explicitly rejected (user call) rather than vendoring mlx-swift a second time. The plan doc's original "training stays mathematically continuous across wakes" goal is **dropped, not silently missed** — doesn't threaten the two headlines, but the loss-curve-continuity secondary deliverable will show real small restart bumps at wake boundaries (expected, annotated in `bg_progress.py`, not a bug). **Registration mechanism corrected during implementation:** SwiftUI's `.backgroundTask` scene modifier only has `.appRefresh`/`.urlSession` cases — no `.processing`, so `BGProcessingTaskRequest` uses the traditional `BGTaskScheduler.register(forTaskWithIdentifier:using:launchHandler:)` API from `LLMEvalApp.init()`, with `expirationHandler`/`setTaskCompleted` driving a cancellable child `Task` (see `LLMEvaluator+BGTrain.swift`'s `handleBGTrainTask`). **Info.plist:** `GENERATE_INFOPLIST_FILE`'s `INFOPLIST_KEY_*` synthesis does NOT support the two array-valued keys needed here (confirmed empty post-build) — fixed via a small merged `Applications/LLMEval/LLMEval-Info-Additions.plist` (`INFOPLIST_FILE` alongside `GENERATE_INFOPLIST_FILE=YES`, which Xcode merges) carrying `UIBackgroundModes=[processing]` + `BGTaskSchedulerPermittedIdentifiers=[mlx.LLMEval.bgtrain]`; confirmed present as real arrays in the built app's Info.plist via `plutil -p`. **Hard cap: 7 calendar days** — if incomplete, report as a headline finding, no foreground fallback. **Monitoring:** daily `devicectl` pull + ad hoc pulls, regenerates a static matplotlib progress figure (`eval/bg_progress.py`, built + smoke-tested against synthetic data) each time. Harness → **h6** (`smollm3-ondevice-train-bg-h6`), new JSONL `train_bench_metrics_e2e_bg.jsonl` + `bg_run_meta.json`. Build verified clean (`xcodebuild` succeeded, device destination). Force-quit-disables-next-wake behavior still unverified — no reason to have hit it during short debug cycles; watch for it during the real run. **Xcode debug-forced validation DONE 2026-07-13** (3 cycles via `_simulateLaunchForTaskWithIdentifier:` on user S=`u00008075`): checkpoint/resume confirmed correct across 3 consecutive wakes (10→20→30 iterations, always resuming from the last checkpoint, never restarting at 0). One real gap was found and fixed mid-validation: `bg_run_meta.json` was only written at a wake's graceful end, so a hard kill mid-wake (Xcode Stop, or a real SIGKILL) left that wake completely absent from the wake-timeline data even though `checkpoint_meta.json`/weights survived fine — fixed by upserting a lightweight partial wake-summary entry after every chunk checkpoint (not just at wake end), keyed by `wakeNumber` so re-triggers can't duplicate entries either. Also observed, unexplained, low-priority: a duplicate `wake_start` on every wake regardless of how many times the LLDB command was actually issued — didn't affect correctness, worth watching during the real run. **Validation data was NOT tagged `validation:true`** (the `--bg-train-validation` launch arg was added to the wrong Xcode scheme — `embedder-tool` instead of `LLMEval`) — remediated by wiping `train_bench_metrics_e2e_bg.jsonl`/`bg_run_meta.json` on-device (zero-byte overwrite) rather than relying on the flag; confirmed self-reset on submission — `bg_checkpoints/u00008075/` was verified empty right after. **Real run submitted 2026-07-13T11:32:10Z** (`--bg-train-submit --user u00008075 --condition bg_overnight`), confirmed via a fresh `bg_train_config.json` pull (`nUser=405`, `iterationsTotal=1215`) and an empty checkpoint dir. One extra gotcha hit at submission time, unrelated to the code: a stale provisioning-profile build error resolved on retry, then the device required a manual "Trust This Developer" tap in Settings ▸ VPN & Device Management before the app would launch at all — both are normal after a fresh install/build-hiccup cycle, not h6-specific. **RESTARTED 2026-07-13.** The first attempt (11:32:10Z) produced ZERO progress across 3 wakes over ~1 hour (11:39:57Z, 11:48:12Z, 12:18:36Z) — no wake ever completed a single 10-iteration chunk (`iterations_completed_total: 0` throughout, no `train`/`checkpoint` records, `bg_run_meta.json` stayed empty), and no further wake fired for 22+ minutes after the third, an unexplained stall on top of the non-progress. Root cause undiagnosed — the original schema had no visibility into whether wakes were dying during model load, LoRA/GC setup, or mid-training-chunk. **Added 3 new persistent JSONL markers** to `runBGTrainWake()` (NOT more `tlog` — tlog's stderr is only readable during a `devicectl --console`/Xcode-attached session, never during a real unattended wake, so more of it wouldn't have helped diagnose this): `model_loaded`, `training_setup_complete` (LoRA/GC applied, resume point known), `chunk_start` (stamped before each `LoRATrain.train` chunk attempt) — a wake that dies now leaves a precise breadcrumb of how far it got. Also added `--bg-train-cancel` (calls `BGTaskScheduler.cancelTaskRequestWithIdentifier:`) as a clean stop lever independent of a full uninstall, for future use. **Full reset performed:** `devicectl device uninstall` (kills any resident process, cancels the pending request, wipes the entire data container in one action — confirmed this is the correct/only reliable way to stop the chain, since `runBGTrainWake()` re-arms the next request as its literal first action, so wiping on-disk files alone would NOT have stopped it from continuing to fire), rebuilt + reinstalled, re-pushed the side-loaded `Documents/user_data/lamp3_*.jsonl` (6 users, from a local cached copy — uninstall wipes this too, easy to miss), resubmitted. **Gotcha:** uninstalling apparently resets the device's per-app trust record, not just app data — needed a second manual "Trust This Developer" tap in Settings before the reinstalled app would launch. **Confirmed clean restart:** fresh `bg_train_config.json` (`submittedAtUtc: 2026-07-13T12:55:52Z`), empty checkpoint dir, JSONL doesn't exist yet (no wake fired since reinstall). Old (voided) telemetry preserved for provenance, not deleted: `results/ondevice/{train_bench_metrics_e2e_bg,bg_run_meta,figures/bg_progress}_2026-07-13_VOIDED-attempt1.*`. **Now monitoring** (on-request, not automated — user pings periodically, no `/loop`/daily-scheduled pulls): pull `train_bench_metrics_e2e_bg.jsonl` + `bg_run_meta.json` via `devicectl device copy from`, save to `results/ondevice/` (dated), run `eval/bg_progress.py`. Do not force-quit the app from the app switcher — confirmed operationally critical, unverified-but-trusted per Apple docs that it disables the next scheduled wake. **Watch closely whether this restart repeats the same zero-progress pattern** — if it does, that itself becomes the headline finding (windows too short for this recipe to ever complete one chunk), and the new diagnostic markers should pinpoint exactly where. **Mistake caught + fixed right after the restart:** `devicectl device uninstall` also wiped the cached ~1.73GB model download (`ageyko/SmolLM3-3B-a1lamp-4bit`), not just training-generated files — CLAUDE.md's documented "model persists across reinstalls" only holds for an ordinary install-over-existing, not a full uninstall. Both submitted `BGProcessingTaskRequest`s have `requiresNetworkConnectivity=false` (valid when the model is cached, not valid for a multi-GB cold download), so a background wake almost certainly couldn't have redownloaded it. Caught because the user opened the app in the foreground to check on it and saw the download in progress; it completed there (foreground Wi-Fi, no background constraints) — confirmed no other state (config/checkpoint dir/pending wake) was disrupted. **Lesson for any future uninstall+reinstall on this project:** re-warm the model cache via a normal foreground app launch before trusting background requests to work — re-pushing side-loaded user data alone isn't sufficient. **First real progress + a second bug found, same day:** wake 0 (13:07:15Z) fully succeeded — model load (10s) → LoRA/GC setup (~33s) → 4 chunks/40 iterations (loss 2.56→2.44→2.72→2.51) → graceful `expiration_handler` termination → checkpoint saved. **First confirmed end-to-end proof the whole mechanism works on a real unattended wake.** But the next 2 wakes (13:19:05Z, 13:28:57Z) both died identically: `model_loaded` fires, then nothing — no setup-complete, no chunk, no checkpoint. The one difference from wake 0: both needed to resume the saved checkpoint. Investigated rather than just adding more markers (explicit user ask): found `loadLoRAWeights` used `Module.update(parameters:)`'s convenience overload, which hardcodes `verify: .none` + `try!` — shape-mismatch checking is skipped entirely, so a mismatch wouldn't throw, it'd silently corrupt state with the real crash surfacing later as an uncatchable native abort (matches the zero-trace symptom exactly). Not confident this is the true cause (identical code worked in debug validation, shapes should be deterministic) — leading hypothesis is memory pressure/jetsam from the extra allocation needed to load+assign the checkpoint on top of an already-loaded model, which background execution has a much tighter budget for than foreground/debug-attached. **Fix (h6 schema v2):** `loadLoRAWeights` now uses the throwing `verify: .shapeMismatch` overload (safety improvement regardless of root cause); added `resume_start`/`resumed` JSONL markers bracketing exactly the weight-load step with a `peak_mem_bytes` reading on success, to directly confirm or rule out the memory-pressure theory next wake. Rebuilt + reinstalled via a normal `devicectl device install` (no uninstall this time) — confirmed the 40-iteration checkpoint, model cache, and side-loaded data all survived intact. **Schema v2 did NOT resolve it:** 2 more wakes (13:59:04Z, 14:29:41Z) died in the SAME narrow window, but neither even reached `resume_start` — pushing localization earlier than the weight-load step itself. Now 4/4 consecutive resume-needing wakes dead in that early region vs. wake 0's lone fresh-start success; genuinely can't yet rule out "later wakes just get shorter OS windows than the lucky-first one" vs. "resume-specific work is the bottleneck" with only one fresh-start data point. **h6 schema v3 (2026-07-13, same day):** added `lora_apply_start`/`lora_apply_complete` markers bracketing `LoRAContainer.from(...)` directly (the one call both paths share), plus a **heartbeat mechanism** — `bg_heartbeat.json`, overwritten ~1x/sec by a concurrent `Task` (started after `wake_start`, cancelled via `defer` on every exit path) independent of milestone markers, so a silent death's last-written `wake_elapsed_s` gives the actual OS-granted time-slice length directly rather than only bounding it between two markers. Rebuilt + reinstalled normally again — checkpoint (40 iters) confirmed intact. **Schema v3's first dead wake (15:00:34Z) gave a much sharper localization**: `lora_apply_start` fired but not `lora_apply_complete`, and the heartbeat's last tick was at wake_elapsed=8.68s (just before `model_loaded`/`lora_apply_start` at ~9.6s) — meaning this wake died within roughly half a second of entering `LoRAContainer.from(...)`, not partway through a slow operation. Since that call is identical regardless of resume state and wake 0 sailed through it fine, this reframed the leading theory away from "resume-specific bottleneck" toward **wildly variable OS-granted window length** (as short as ~10s, vs. wake 0's 300+s). **User's thermal-throttling theory checked against the full trajectory**: thermal_state was `nominal` (fully cooled) at the start of every one of the 4 fast-dying wakes; only the wake immediately following wake 0's actual training ran `fair` (residual heat). Battery pinned at 100%/charging throughout. Data argues against thermal throttling as the driver, though `thermalState` is a coarse 4-level enum with no reading captured at the literal moment of death (wakes die too fast to log one) — not a hard rule-out, but doesn't support the theory either. **Deep research done (WebSearch, ~9 queries) on whether BGProcessingTask variance is a known phenomenon and whether better alternatives exist:** Apple's own `BGProcessingTaskRequest` header hedges explicitly — grants are "best-effort... as long as the user has used your app within the past week"; multiple Apple Developer Forum threads since 2020 report tasks suspended after as little as ~5 seconds despite "several minutes" being the documented target for heavy work (Core ML training is Apple's own cited example use case), and describe BGProcessingTask as "only working a fraction of the time" — closely matching our own empirical distribution (10s to 300+s). One forum lead, not confirmed: sustained high CPU utilization during a granted window can itself trigger an early watchdog kill despite BGProcessingTask nominally relaxing CPU limits — plausible and directly relevant given LoRA training is CPU/GPU-heavy, but unconfirmed by Apple docs. iOS explicitly runs energy/data budgets across the day and favors frequently-used apps in scheduling — our test methodology (submit once via `devicectl`, then leave the phone alone) doesn't resemble "frequent engagement" to the predictive engine, a real external-validity caveat: genuine production usage might score better than this unattended test does. Confirmed iOS 26's new `BGContinuedProcessingTaskRequest` (foreground-initiated, user-visible progress UI, possibly gated behind a background-GPU entitlement of uncertain availability on a free personal team) exists seemingly to address exactly this gap — read as corroboration that `BGProcessingTask` is the wrong tool for sustained heavy compute, not evidence of a bug in our implementation. Found a legitimate academic precedent for the empirical-measurement methodology itself: [Chen et al., "Smartphone Background Activities in the Wild," MobiCom 2015](https://www.sigmobile.org/mobicom/2015/papers/p40-chenA.pdf) (large-scale background-activity measurement across thousands of phones; pre-dates `BGTaskScheduler` but same general phenomenon) — no peer-reviewed paper found specifically reverse-engineering modern `BGProcessingTask` heuristics, which if anything makes this round's own characterization more novel. **Decision: keep the current `BGProcessingTask` round running as designed, do NOT pivot to `BGContinuedProcessingTask`** — that API answers a materially different research question (foreground-initiated + continues-into-background, not silent OS-scheduled wakes) and would be a new round, not a fix to this one. Treat "windows are highly variable and often too short for this recipe" as a real, well-corroborated headline finding for co-headline #2, citing the above sources as context in the eventual write-up (`experiments/2026-07-2X-ondevice-bg-training.md`). **2026-07-14: the ~30min/~10s pattern held for 30+ wakes over 16h then broke** (a ~4.5h gap, then a wake dying even earlier than usual, before `model_loaded`). Built `eval/bg_timeslice.py` (new: x=time since launch, y=granted time slice per wake, log scale) for fast repeated check-ins. **Root-caused rather than just observed further, per user push:** every wake since wake 0 dies SILENTLY (no `wake_end`, so `setTaskCompleted` never fires) — the only `Task.isCancelled` check was inside the training while-loop, never earlier, e.g. before the synchronous `LoRAContainer.from(...)` call where every recent death occurs. Hypothesis (explicitly unconfirmed, undocumented iOS internals): the scheduler may track clean-completion-acknowledgment as a signal for grant *size* (not scheduling *frequency*, which stayed rock-steady throughout — a distinction the user correctly pushed me to sharpen re: what `setTaskCompleted(success:)` actually means). **Fix (h6 schema v4):** `Task.isCancelled` checks added at every setup-path step boundary (not just the while-loop) + an independent loose wall-clock backstop (`bgWallClockBackstopS=25.0`, well above the ~10s death zone so it won't preempt a long wake like wake 0's ~324s) — both trigger a clean `return` so `setTaskCompleted` fires even on a zero-progress wake. Added `--bg-train-resubmit` (re-arms without wiping config/checkpoint/cap-origin, unlike `--bg-train-submit`). Cancelled the old request, rebuilt, normal-reinstalled (checkpoint 40 iters + config timestamp both confirmed intact), resubmitted. Full narrative + reasoning in memory (`project_bg_ondevice_training_plan.md`) — kept this entry to a summary to avoid further bloat. **v4 outcome (2026-07-14):** the wall-clock backstop DID fire cleanly once (a wake suspended by the OS for ~30.7 min, caught right after `model_loaded` with `wake_termination_reason: wall_clock_backstop`) — first proof the mechanism works — but the very next wake reverted to the same ~10s silent death, and across 5 more wakes checked the following morning, still 40/1215 iterations, zero grant-size recovery visible. **h6 schema v5 (2026-07-14):** `runBGTrainWake()` used to loop, attempting as many `bgChunkIterations`-sized chunks as time allowed within one wake (wake 0 ran 4 back-to-back before being cancelled) — i.e. always grabbing as much as it could get. Changed to attempt exactly ONE chunk per wake, checkpoint, then voluntarily return (`voluntary_yield`) even if more time was clearly available, layered on top of the unchanged v4 safety net. Explicit hypothesis test (user-proposed), not a confirmed fix: does a consistently small, quick, always-completes-cleanly request pattern earn steadier scheduling than a greedy one? Doesn't address the current dominant failure mode (dying during model load/LoRA setup, before any chunk is ever reached) — a complementary, lower-priority experiment. Cancelled the pending request, rebuilt + normal-reinstalled (checkpoint 40 iters confirmed intact via a fresh pull), resubmitted via `--bg-train-resubmit` (preserves checkpoint + original 7-day cap origin). Only 1 wake fired under v5 before the next step (below) — died silently at ~9.5s, same as the dominant pattern, never reaching the chunk loop (as flagged, v5 can't matter until a wake gets past setup). **TRUE RESTART 2026-07-14T16:09:08Z** (user request, after noticing `--bg-train-resubmit`'s by-design checkpoint/cap-origin preservation meant totals still read 40/1215 and "26.2h since launch" despite the v5 rebuild): cancelled, zero-byte-wiped `bg_run_meta.json`/`bg_heartbeat.json`/`train_bench_metrics_e2e_bg.jsonl`/`bg_train_config.json` on-device via `devicectl device copy to`, then `--bg-train-submit --user u00008075 --condition bg_overnight` (wipes the checkpoint dir itself + writes a fresh config). Confirmed clean: new `bg_train_config.json` (`submittedAtUtc: 2026-07-14T16:09:08Z`, `nUser=405`, `iterationsTotal=1215`), empty checkpoint dir, empty run-meta, 0-line JSONL. **New cap: 2026-07-21T16:09:08Z.** All prior telemetry (wakes 0-44, schema v1-v4 plus the one v5 wake) preserved locally as dated pulls in `results/ondevice/` — genuinely a fresh 0/1215 run under schema v5 now, not a resumed one. **7/7 post-restart wakes all died at 9.3-9.9s (median 9.5s)** — a tight, repeatable ceiling, not noisy variance. Checked all documented gating factors: Background App Refresh ON (global + per-app), Low Power Mode OFF, thermal nominal, entitlements/Info.plist confirmed correct (submit never throws `BGTaskSchedulerErrorCodeNotPermitted`). Corroborated by extensive external research (Apple dev forum threads going back to iOS 13, Apple's own guidance "never rely on background tasks for core functionality") that BGProcessingTask grant-size variance/short windows are a widely-reported, multi-year, cross-app phenomenon, not specific to this project. **Control experiment added 2026-07-15 (user request):** `ios/BGProbe/` — a brand-new, standalone, minimal Xcode project (NOT a target on mlx-swift-examples.xcodeproj, to keep zero risk to the live h6 round), bundle id `com.geyko.bgprobe`, task id `mlx.bgprobe.test`. Does nothing but register + resubmit a `BGProcessingTaskRequest` and log a 0.2s-resolution heartbeat (`Documents/bgprobe.jsonl`) — no model load, no heavy allocation, no real CPU/GPU work. Tests directly whether the ~9.5s ceiling is specific to LLMEval's resource footprint (model load + LoRA/GC setup) or a platform/device-level constraint that also hits a trivial, freshly-installed app with no prior scheduling history. Installed + launched once (foreground) to trigger initial registration; `submit: ok=true` confirmed via a fresh `bgprobe.jsonl` pull. **RESULT (2026-07-15): footprint hypothesis confirmed.** BGProbe's first 5 wakes: 57.4s (real graceful OS expiration), then 240.1-240.2s ×4 (hit BGProbe's own defensive 240s cap — the OS never cut it off; true ceiling is at least 4+ min). Directly compared against LLMEval in the same window: **wake timestamps matched EXACTLY between the two apps** (07:50:43Z, 08:20:54Z, 08:51:53Z, 09:24:47Z, down to the second — the OS batches both apps' background opportunities into the same maintenance windows) — LLMEval got ~9.5-10.2s at every one of those same instants; BGProbe got 240s+. **This is NOT a platform/OS-level ceiling** — the OS is willing to grant several minutes on this exact device/iOS build/moment, just not to LLMEval specifically. Proximate cause is almost certainly LLMEval's own resource footprint (loading the ~1.7GB 4-bit model) triggering an early memory-pressure kill right around `lora_apply_start`, not a generic short-grant phenomenon. Reopens "shrink LLMEval's setup footprint" as the clearly-indicated next step. **h6 schema v6 (2026-07-15, diagnostic-only, no behavior change):** added `peak_mem_bytes`/`active_mem_bytes` (`Memory.snapshot()`) to the `model_loaded`/`lora_apply_start`/`lora_apply_complete` markers, a single `GPU.resetPeakMemory()` near wake start for a clean per-wake baseline, and bumped the heartbeat cadence 1s→200ms (also now carrying the same memory fields) — prior localization already pinned death to within ~0.1-0.5s of `lora_apply_start`, too fast for 1s heartbeat resolution to say more; this should pin both timing and memory footprint at the moment of death much more precisely. Cancelled pending request, rebuilt (`BUILD SUCCEEDED`), normal-reinstalled (checkpoint dir confirmed genuinely empty — 0/1215 iterations completed since the 2026-07-14 restart, no wake has ever finished a chunk — and config's original `submittedAtUtc` intact), resubmitted via `--bg-train-resubmit`. Watching next wakes for the finer death localization + memory readings.
- **LaMP coverage expansion (R7, queued) — PRE-EXECUTION, not yet started.** Design doc: `experiments/2026-07-10-lamp-coverage-expansion-r7-a2lamp-plan.md`, pinned 2026-07-10 via `/grill_me`. Extends the LaMP suite from {3,4,7} to all six publicly-downloadable LaMP tasks — adds LaMP-1 (citation ID), LaMP-2 (**both** variants: movie-tagging current + news-categorization deprecated, confirmed both hosted at `LaMP_2/new/` and `LaMP_2/` respectively), LaMP-5 (scholarly title gen). LaMP-6 stays excluded (private Avocado corpus). Cluster-side only, runs **in parallel** with Phase 3 below (different hardware, no conflict; Phase 3 stays PRIMARY). **Key architecture decision (reversed once mid-design, see doc's "Decisions that reversed mid-grill"): ONE shared Task-LoRA, not per-task adapters.** R7 = harness engineering (new `"classification"` metric path in `eval_lamp.py`: Accuracy+macro-F1 for LaMP-1/2) + full retrain of a new adapter **A2-lamp** from the frozen base on all 7 tasks' training data concatenated (same recipe as A1-lamp, bigger corpus) + BFCL check + a 7-task baseline/BM25/FT table. **A2-lamp will replace A1-lamp as the canonical Task-LoRA** once it lands — A1-lamp stays on disk untouched, marked historical, not deleted. Because the canonical base changes, R5 (LaMP-3 User-LoRA) and R6 (LaMP-4 User-LoRA) are slated for re-run stacked on A2-lamp in a later round (R8+), and the Llama-3.1-8B/70B scale comparison is slated for re-run/extension to all 7 tasks — neither has started. New User-LoRA rounds for the 4 new tasks are also queued (R10-R13ish) but not yet designed; per-round design docs get written once R7's real numbers exist, not before. Full round-by-round scope table is in the design doc — read it before touching any of this.
- **Phase 3 — E2E on-device per-user training (PRIMARY).** Status **IN PROGRESS as of 2026-07-13** (corrected — was stale-documented as "PRE-EXECUTION," actually provisioned and running since ~2026-07-06). Plan locked (grilled): `experiments/2026-07-03-ondevice-e2e-training-plan.md`. Shift from *cost benchmark* (generic 50-line data, weights discarded, fixed 200 steps) → *E2E*: train **real top-100 LaMP-3 User-LoRAs to completion (3 epochs) with the faithful R5 recipe, save the adapters**, and measure cost + how it degrades under adverse conditions. **Primary claim = COST** (time/energy/thermal/memory) as a function of profile size → extrapolate to all 100; **secondary = FIDELITY** via train-loss overlay vs cluster R5 (accuracy only a nice-to-have). **Recipe (minimal edits, NO fork):** 4-bit SmolLM3 **+ fused A1-lamp Task-LoRA**, `AdamW(1e-5)` (default wd 0.01 == R5 L2), r=8 q+v α16, batch 1, GC on (h4), `iterations=3×n_user`, cap 1024, save adapter. Forced deviations: 4-bit not bf16, no dropout, batch 1 not eff-8, fixed LR not cosine. **6 sample users**: S=`u00008075`/405, M=`u00005020`/550, L=`u00012502`/987, 448=`u00005228`, 500=`u00011077`, 653=`u00013218`. **4 conditions:** C0 ideal / C1 Low-Power-Mode / C2 unplugged / C4 heavy-3D-game contention. **14-run matrix**: S C0×1+C2×2 (3), M C0×3+C1×2+C4×2 (7), 448/500/653/L C0×1 each (4) = 14 total, ~25 device-hrs. **5 plots** (cost-vs-profile+extrapolation, thermal trajectory, condition bars, loss overlay, battery drain). **Energy only measurable UNPLUGGED** (no per-process power API) → C2 on short user S. Harness → **h5** (`smollm3-ondevice-train-e2e-h5`), JSONL `train_bench_metrics_e2e.jsonl` (dated pulls in `results/ondevice/`, cumulative — latest pull is authoritative, aggregate via `eval/e2e_aggregate.py`). **Progress as of the 2026-07-08 pull (7/14 runs done):** S C0×1 ✅, S C2×1/2 (1 more C2 needed), M C0×2/3 + C1×0/2 + C4×0/2 (5 more needed), 448 C0×1 ✅, 500 C0×1 ✅, L C0×1 ✅, **653 (`u00013218`) not started at all**. **Remaining 7 runs: S C2×1, M C0×1+C1×2+C4×2, 653 C0×1.** **BLOCKED until the h6 background-training round (above) concludes** (completes or hits its 2026-07-20 cap) — same phone, same app; a rebuild/reinstall to run more E2E launches risks disrupting the live pending `BGProcessingTaskRequest`/checkpoint, and more importantly, active foreground use of the app would itself feed iOS's engagement-recency scheduling heuristic and bias the h6 round's "unattended background" measurement (see h6 entry — confirmed via research, not speculative). Provisioning (fuse+convert 4-bit, publish HF repo `SmolLM3-3B-a1lamp-4bit`, side-load 6 user JSONL) is long done, not a blocker; see plan §Provisioning if ever re-needed from scratch.
- **Phase 3 — on-device *cost* training benchmarks (DONE, superseded by E2E above).** Naive baseline **DONE 2026-06-29** (`experiments/2026-06-29-ondevice-training-naive.md`): naive LoRA FT of SmolLM3-3B-4bit jetsams on the first backward step at deployment seq lengths; feasible only to a **256-tok ceiling** (cap=512 OOMs), throttled there (0.41 iter/s, 4.1 GB). **Gradient-checkpointing variant DONE 2026-06-30** (`experiments/2026-06-30-ondevice-training-gc.md`). **Headline:** per-block GC lifts the ceiling **256 → 1024 tok (4×)** — cap=512 AND cap=1024 now train, zero OOM across the full sweep; memory savings grow with seq len (−17% @32 → −41% @256); **GC@1024 (4014 MB) fits in less peak than naive@256 (4128 MB)**; recompute cost ~0.78–0.85× naive iter/s. The bound is now thermal, not memory (cap=1024 = ~52 min/200 steps, `serious`). **Implementation:** per-block checkpoint via public MLX `CustomFunction`+`vjp` (NOT the raw `mlx_checkpoint` C binding — `Cmlx` isn't a public product; the C route would force vendoring mlx-swift too), LoRA params threaded as explicit differentiable inputs. `mlx-swift-lm` is now **vendored as a local SPM override** at `ios/mlx-swift-lm-local/` (replaces the remote pin in `project.pbxproj`; edits confined to `Libraries/MLXLLM/Models/SmolLM3.swift` — `SmolLM3Model.useGradientCheckpoint` flag). No fork of `LoraTrain.swift` needed (flag on the model drives the stock trainer). Harness h4, separate JSONL `train_bench_metrics_gc.jsonl`. (The "stack next MeBP technique" idea is deferred behind the E2E run.)
- **Phase 3 — base-vs-Task-LoRA inference.** Deferred (was next milestone before training track opened). Also same-phone/same-app as the E2E and h6 rounds above — same "blocked until h6 concludes" reasoning applies once actually picked up. When resumed: fuse A1-lamp ckpt-1000, convert to MLX, swap `modelConfiguration` id, measure with existing inference rig.
- **Round 6 (LaMP-4 multi-user) — DONE 2026-07-07.** Writeup: `experiments/2026-07-07-user-lora-lamp4-round6-multi.md`. **Headline:** cross-task OPPU-recipe replication on LaMP-4, K=100 users. Mean per-user ROUGE-1 0.235→0.242 (Δ+0.007), **not statistically significant** (paired-t p=0.20, Wilcoxon p=0.30, 95% CI [−0.002, +0.018] spans zero) — falls in the plan's pre-specified "≈0" disposition bucket, consistent with OPPU's own weak LaMP-4 result (+0.003 R-1) and our four single-user LaMP-4 rounds (R1–R4, all sub-MDE). Expected outcome per the plan's own priors, not a setback — R5's LaMP-3 confirmation of Q4 is unaffected; no R7 follow-up queued. Paper table updated: `overleaf/6a2b1ada3ba0566171e752a2/sections/experiments/2026-06-18-per-user-lora-lamp3.tex` (`tab:r6-lamp4-multi`). Two infra issues surfaced and fixed along the way (see writeup): `tyr1` GPU-slot oversubscription (retry, no code change) and a Blackwell (sm_120) incompatibility on `fornjoter` (now guarded via `require_gpus` on all LaMP-4 GPU subs — the same guard the Llama subs already had; CLAUDE.md's own GPU-capability-ceiling note had already flagged this as likely to eventually hit un-hardened subs).
- **Llama-family scale comparison — DONE 2026-06-30.** Writeup: `experiments/2026-06-30-llama-scale-comparison.md`. Headline: **SmolLM3-3B + A1-lamp Task-LoRA beats Llama-3.1-70B-Instruct + BM25 on all three LaMP tasks** (LaMP-3 +0.006, LaMP-4 +0.017, LaMP-7 +0.116). On the K=100 personalization-hard subset, the full two-LoRA stack (Task + User) goes further: acc 0.730 / MAE 0.290 vs Llama-70B+BM25 0.700 / 0.330. Scale alone narrows but does not close the gap to fine-tuning. Aggregator: `eval/tables.py`. Caveats: point estimates only (no CIs); R5's User-LoRA lift is at MDE (p≈0.10), so Table 2 inherits that.

**The "no on-device/mobile code" hard constraint is LIFTED for Phase 3** (it
remains the historical framing for Phases 1–2).

---

## Phase 3 runbook — on-device deployment

### Runtime decision (verified against official sources only)

- **Runtime = MLX** (`mlx-swift` on device, `mlx-lm` on the Mac). llama.cpp / ExecuTorch rejected.
- Apple Foundation Models is the standard framework; "Core AI" is its ship-your-own-local-model provider. Apple's adapter-training toolkit is Apple-model-only (rank-32 LoRA bound to the OS system model) — cannot adapt SmolLM3.
- **SmolLM3 is first-class in MLX**: `mlx_lm/models/smollm3.py` (Python) and `Libraries/MLXLLM/Models/SmolLM3.swift` (Swift, in `ml-explore/mlx-swift-lm`). Confirmed by reading source.
- MLX is also the credible **on-device training** route (`mlx_lm.lora`, the `LoRATrainingExample` app, Apple paper arXiv:2510.03425).
- **Model delivery = HF download on-device.** The app pulls `mlx-community/SmolLM3-3B-4bit` into its sandbox on first generation. We publish our own HF repo only once we fuse Task-LoRA.

### Host / device / signing facts

- Mac: Apple **M3, 16 GB**. Full Xcode 26.5 at `/Applications/Xcode.app`, but active dev dir is CommandLineTools — **prefix every Xcode/devicectl command** with `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Signing: **Apple Development: andrew.geyko@icloud.com**, team **`JGW9U9Y36Y`** (free personal team; bundle IDs auto-disambiguated via `DISAMBIGUATOR=${DEVELOPMENT_TEAM}` in `Configuration/Build.xcconfig`).
- Testbed: **iPhone 17 Pro** (`iPhone18,1`), iOS **26.5.1**, Developer Mode on. UDID **`00008150-000674C60A3B401C`**. List: `xcrun devicectl list devices`.

### Local MLX toolchain (Mac-side)

- venv **`.venv-mlx/`** (Python **3.11** — 3.14 too new for MLX wheels), `mlx-lm` (mlx 0.31.2). Gitignored.
- Convert + quantize: `.venv-mlx/bin/python -m mlx_lm convert --hf-path HuggingFaceTB/SmolLM3-3B --mlx-path data/models/SmolLM3-3B-mlx-4bit -q --q-bits 4`
- Sanity generate: `.venv-mlx/bin/python -m mlx_lm generate --model data/models/SmolLM3-3B-mlx-4bit --prompt "..." --max-tokens 60` (SmolLM3 has **thinking mode on by default** — emits `<think>…</think>`).

### iOS app — vendored, edited, built, deployed

- **`ios/mlx-swift-examples/`** is vendored into this repo via `git subtree` (upstream `ml-explore/mlx-swift-examples` base `378f244`). Edit harness files → ordinary `git commit`. Bump upstream: `git subtree pull --prefix=ios/mlx-swift-examples https://github.com/ml-explore/mlx-swift-examples <tag> --squash`. Only `build/` + Xcode user state gitignored. See `ios/README.md`.
- LLM libs come from `ml-explore/mlx-swift-lm`, **vendored as a LOCAL SPM override** at `ios/mlx-swift-lm-local/` (was remote-pinned; converted 2026-06-30 for the gradient-checkpointing experiment — `SmolLM3.swift` has the per-block GC support). The Xcode project references it via `XCLocalSwiftPackageReference "../mlx-swift-lm-local"`; `.build/` gitignored, source tracked. To bump upstream, re-copy a fresh checkout (minus `.build`/`.git`) over the local dir and re-apply the SmolLM3 GC edits. mlx-swift itself is still remote-pinned. mlx-swift-examples remains git-subtree vendored.
- **Edited:** `ios/mlx-swift-examples/Applications/LLMEval/ViewModels/LLMEvaluator.swift` (`modelConfiguration` → `mlx-community/SmolLM3-3B-4bit`; `appendBenchRecord(...)` appends one flat-JSON line per generation to `Documents/bench_metrics.jsonl`; `hardwareModelIdentifier()` helper).
- Benchmark harness: `ios/mlx-swift-examples/Applications/LLMEval/Benchmark/{BenchmarkSupport,LLMEvaluator+Benchmark}.swift` (+ edits to `LLMEvaluator.swift`, `ContentView.swift`, `project.pbxproj`). Bump `BenchConstants.appBuild` whenever harness logic changes.

**Build (device, signed)** — `-skipMacroValidation` is **required** (else fails on `MLXHuggingFaceMacros … must be enabled`):
```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination 'id=00008150-000674C60A3B401C' \
  -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=JGW9U9Y36Y build
```
Output: `build/Build/Products/Debug-iphoneos/LLMEval.app`, bundle id **`mlx.LLMEvalJGW9U9Y36Y`**.

**Install + launch:**
```
xcrun devicectl device install app --device 00008150-000674C60A3B401C build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y
```
First generation downloads ~1.73 GB from HF (Wi-Fi, one-time; model persists across reinstalls of the same bundle id). The app UI requires a human tap to start generation.

### Reading metrics off the device

No live console (macOS `log stream` has no `--device`; `log collect --device` needs root; `idevicesyslog` not installed). Pull the JSONL from the app container (no sudo); accumulates one line per run:
```
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl --destination /tmp/devpull/bench_metrics.jsonl
```

### Base-inference benchmark — DONE 2026-06-21

Design: `experiments/2026-06-21-ondevice-base-inference-plan.md`. Writeup: `experiments/2026-06-21-ondevice-base-inference.md`. Rig is reused verbatim for base-vs-Task-LoRA.

**Headline (steady-state, cool device, n=5/cell):** decode ~37 tok/s at deployment context sizes (gen64: 38.8 @64-tok prompt → 32.0 @2048), prefill 620–740 tok/s, cold app-launch→first-answer ≈ 1.7 s (model load 1362±114 ms + cold TTFT 380±6 ms), realistic LaMP-3 (natural EOS) 35.0±1.2 tok/s, peak ≈ 2.2 GB (at 2048-tok contexts).

**Two findings that shape the next pass:**
1. **Sustained decode throttles −53%** (37.8 → 17.9 tok/s over 5 min / 6144 tokens, knee ~90 s) and `ProcessInfo.thermalState` stayed `nominal` throughout — coarse enum is useless as throttle proxy; trust per-segment tok/s.
2. **Clean steady-state long-decode curves unobtainable while plugged.** Pre-registered unplugged-over-Wi-Fi follow-up is the way to get a clean decode curve — deferred, not blocking.

**Harness verified vs `mlx-swift-lm` source:** EOS suppression for forced length = drive `TokenIterator` directly and ignore the stop-token set to `maxTokens` (EOS check is in MLXLMCommon's loop wrapper, not `TokenIterator.next()`); `model_load_ms` brackets the `ModelContainer` load. Launch args: `--benchmark` (cold + prefill + decode), `--benchmark-tail` (realistic + 5-min stress, separate launch), `--benchmark-cold` (load+1 gen+exit, ×3 for cold variance). `app_build` baked in (`smollm3-ondevice-bench-h2`).

**Deliverables:** aggregator `eval/bench_aggregate.py` (stdlib-only); raw telemetry `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl` (89 records); aggregate `results/ondevice_base_smollm3_4bit_2026-06-21.json`.

**`peak_mem_bytes` caveat:** MLX `peakMemory` is a process high-water mark (monotonic within a session), so per-cell peaks are confounded by execution order — meaningful number is the session peak (~2.2 GB). Reset-per-run is the h3 improvement for the base-vs-LoRA comparison.

### 7B-class characterization (Qwen3-8B-4bit) — DONE 2026-06-22

Exploratory "does the next size class up fit + run, and at what cost" pass. Reuses the 3B plan verbatim (same grid/regime/harness), only the subject model swapped. Writeup: `experiments/2026-06-22-ondevice-qwen3-8b-inference.md`. Subject = `mlx-community/Qwen3-8B-4bit` (8.2B; first-class `qwen3` arch in `mlx-swift-lm`; `enable_thinking:false` matches the SmolLM3 thinking-off regime). **`peak_mem_bytes` caveat above is RESOLVED here** — harness bumped to **h3** (`app_build` `qwen3-8b-ondevice-bench-h3`): `GPU.resetPeakMemory()` before each measured gen → clean **per-cell** peak; `git_commit`/`git_dirty` now baked into each record. `--benchmark` (`.full`) ran the whole suite (cold+prefill+decode+realistic+stress) in one ~95-min session.

**Headline — feasibility YES:** Qwen3-8B-4bit runs on the iPhone 17 Pro, **no OOM/jetsam**, per-cell peak **4.74 GB (p64) → 5.43 GB (p2048)** under the `increased-memory-limit` entitlement. Cost vs the 3B (clean nominal channel = cold + prefill-64): **decode ~15.5 vs ~37 tok/s (0.40×), prefill ~240 vs ~684 tok/s, cold app-launch→answer ≈ 2.8 vs 1.7 s (load 2043 ms + TTFT 770 ms), peak ~5.0 vs ~2.2 GB** — all tracking the ~2.7× param ratio. Realistic LaMP-3 tier = clean 1-token answers (thinking-off confirmed for Qwen3).

**Two findings:**
1. **8B heat-soaks the phone fast** — reaches `serious` *within* the prefill sweep; sustained/decode collapses to ~5 tok/s (a 2048-tok answer can take >7 min). Plugged-and-idle it can't recover between cells. 8B deployment is thermally bounded, not throughput-bounded.
2. **At 8B `ProcessInfo.thermalState` DOES report the throttle** (`serious`) — opposite of the 3B, where the enum lied `nominal`. The load is heavy enough the coarse enum catches it; still trust per-segment tok/s as primary.

Telemetry `results/ondevice/bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl` (68 records); aggregate `results/ondevice_base_qwen3_8b_4bit_2026-06-22.json`. Decode/realistic/stress cells are all hot-device (`serious`) — clean steady-state 8B decode curve still needs the deferred unplugged-over-Wi-Fi run. `modelConfiguration` id is left at `mlx-community/SmolLM3-3B-4bit` (the training-track default); flip the one line in `LLMEvaluator.swift` to Qwen3-8B for 8B work.

### Capped-stress (bursty-workload) throttle — DONE 2026-07-03

Writeup: `experiments/2026-07-03-ondevice-capped-stress.md`. Re-ran the sustained-stress test as a **realistic bursty workload**: repeated **128-tok** forced generations back-to-back for **10 min** (each a fresh 256-tok prefill), device cooled to `nominal` first — instead of one continuous 60k-tok decode. New harness mode `--benchmark-stress-capped` (`runStressCapped`), harness **h4** (`ondevice-bench-stresscap-h4`, schema v4), one JSONL record per generation with a real-wall-clock `stress_elapsed_s` column. Aggregator (`bench_aggregate.py`) stress block now carries `elapsed_s` + stress `peak_mem_bytes`; plot (`plot_thermal_stress.py`) prefers real elapsed + has a `--title` flag.

**Headline:** bursty throttles nearly as hard as continuous. **3B 38.5→21.0 tok/s (−46 %)**, knee ~83 s; **8B 16.0→9.5 tok/s (−40 %)**, both nominal→fair→serious, no OOM/jetsam, **flat peak 2.11 / 5.01 GB**. Per-query prefill gaps buy a little headroom (8B settles ~9.5 vs continuous ~5) but don't avoid the throttle — **the steady-state budget a user feels under sustained use is the plateau (~half the cold-decode rate), not the cold-start number**. 3B stays interactive throttled; 8B marginal (~13.5 s / 128-tok answer hot). Telemetry `results/ondevice/bench_metrics_{smollm3-4bit,qwen3-8b-4bit}-stresscap_2026-07-03.jsonl`; aggregates `results/ondevice_stresscap_{smollm3_4bit,qwen3_8b_4bit}_2026-07-03.json`; figures `results/ondevice/figures/capped_stress_*_2026-07-03.{pdf,png}`.

**Gotcha:** the 8B run hung ~25 min at model load — Qwen3-8B-4bit had been evicted from the app sandbox by reinstalls since 2026-06-22, so first load re-downloaded ~4.3 GB and stalled. `devicectl … process signal --signal SIGKILL` on the stuck PID + relaunch recovered it. **Before any 8B on-device launch after a gap, expect a re-download; if it hangs (cold phone, no JSONL growth), kill + relaunch.**

**Paper:** this replaced the sustained-decode figure in the write-up. The LaTeX lives in a **separate git repo at `~/Documents/Research/overleaf/`** (remote `git.overleaf.com`, `git pull`/`git push` to sync). The on-device inference experiment is `sections/experiments/2026-06-21-ondevice-base-inference.tex`; its `\autoref{fig:thermal-stress}` now includes `sections/figures/thermal_stress_overlay.pdf` = the bursty capped-stress overlay (copied from `results/ondevice/figures/capped_stress_overlay_2026-07-03.pdf`), stress paragraph + caption updated to match. Pull before editing, push when done.

### Phase 3 next steps

- **E2E on-device per-user training (PRIMARY next):** execute `experiments/2026-07-03-ondevice-e2e-training-plan.md`. Provisioning first (pull A1-lamp ckpt + raw `lamp_time/LaMP_3` + R5 per-user loss from cluster; fuse+convert 4-bit; publish HF `SmolLM3-3B-a1lamp-4bit`; side-load 6 user JSONL). Then harness h5 (`AdamW`, `iterations=3×n_user`, save adapter, capture loss, timed battery, `--user` arg). Then the 14-run matrix. Note: the fuse+convert+publish step here is the SAME artifact the base-vs-Task-LoRA inference milestone needs — do it once.
- **Task-LoRA on-device inference (base-vs-Task-LoRA):** reuses the fused HF model published above; swap `modelConfiguration` id, measure base vs Task-LoRA with the inference rig.
- **Unplugged decode-curve follow-up** (pre-registered, deferred).

---

## Phases 1 & 2 — frozen state

### Research questions

| Q | Status |
|---|---|
| **Q1** — Does fine-tuning on LaMP help a 3B model at all? | **YES** — A1-lamp ckpt-1000 gives +0.11 / +0.07 / +0.13 on LaMP-3/4/7 test over BM25 baseline. |
| Q2 — Synthetic preference-conditional data on top of LaMP? | DROPPED with 2026-06-02 pivot. |
| Q3 — General Task-LoRA vs domain-specific? | DROPPED with 2026-06-02 pivot. |
| **Q4** — Per-user LoRA on time-ordered user history beyond Task-LoRA alone? | **YES (LaMP-3)** confirmed by R5 (2026-06-19): ΔMAE −0.050, acc 0.680→0.730, RMSE 0.616→0.575, zero inference overhead. **Does not extend to LaMP-4**: R6 (2026-07-07) found mean ΔR-1 +0.007, not significant (p=0.20) — expected per the plan's own priors (OPPU's own LaMP-4 lift is a weak +0.003 R-1). Q4 stands as LaMP-3-specific. |
| **Q5** — Does the 3B + two-LoRA stack survive a scale comparator (Llama-3.1-{8B,70B}-Instruct + BM25)? | **YES** (2026-06-30): A1-lamp Task-LoRA beats Llama-70B+BM25 on LaMP-3/4/7 by +0.006/+0.017/+0.116; on K=100 LaMP-3 the two-LoRA stack reaches 0.730 acc / 0.290 MAE vs Llama-70B+BM25 0.700 / 0.330. `experiments/2026-06-30-llama-scale-comparison.md`. |

### Canonical artifacts

- **A1-lamp Task-LoRA:** `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/` (1-epoch sweep, step 1000, epoch 0.75, frozen 2026-06-02).
- **100 LaMP-3 User-LoRAs (R5):** `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/` (one per user in `data/lamp_user_stats/LaMP_3_top100_users.json`).
- **4 single-user LaMP-4 User-LoRAs (R1/R2-B/R4):** retained for re-analysis.
- **100 LaMP-4 multi-user User-LoRAs (R6):** `train/checkpoints/user_lora_lamp4_<fp>_oppu_seed0/final/` (one per user in `data/lamp_user_stats/LaMP_4_top100_users.json`).

### Phase 1 headline numbers (LaMP test, seed=0, greedy, BM25 k=4 — dev numbers within ±0.01; see `experiments/2026-06-13-lamp-test-split-correction.md`)

| Task | No-profile floor | Profile baseline | A1-lamp (ckpt-1000) | Δ adapter − baseline |
|---|---|---|---|---|
| LaMP-3 (acc) | 0.4508 | 0.6964 | **0.8056** | **+0.109** |
| LaMP-4 (rouge1) | 0.1393 | 0.1537 | **0.2259** | **+0.072** |
| LaMP-7 (rouge1) | 0.4170 | 0.4372 | **0.5619** | **+0.125** |
| BFCL AST overall | — | **0.8078** (Py-only 0.8870) | **0.7696** | **−0.038** |

Result files: `results/LaMP_{3,4,7}_test_a1_lamp_1ep_seed0_checkpoint-1000_bm25k4_seed0.{json,predictions.jsonl}` (test, canonical), plus `_dev_*` variants. BFCL: `results/bfcl_ast_a1_lamp_1ep_seed0_checkpoint-1000_seed0.{json,predictions.jsonl}`. Baselines: `results/LaMP_{3,4,7}_test_base_{bm25k4,noprofile}_seed0.*` and `results/bfcl_ast_base_seed0.*`.

Earlier `a1_lamp_seed0/` (2-epoch run) is Pareto-dominated but on disk for provenance. Full Pareto sweep narrative: `experiments/2026-06-02-a1-lamp-1ep-pareto.md`. `checkpoint-400` is the alternative if maximum BFCL retention is the dominant criterion.

### Phase 2 history (one line per round)

Single-user u00000011 LaMP-4 rounds **R1–R4 all failed pre-registered gates on test** (dev/test asymmetry across all four: dev Δ +0.030/+0.043/+0.047/+0.040 vs test Δ +0.003/−0.004/+0.010/−0.018). **R5 LaMP-3 K=100 OPPU recipe** (r=8 q+v only, LR=1e-5, L2=1e-2, 3 epochs, stacked on A1-lamp ckpt-1000) confirmed Q4 at MDE. **Phase 2 closed 2026-06-19**, reopened same day as R6 cross-task descriptive replication. **R6 done 2026-07-07**: LaMP-4 replication, mean ΔR-1 +0.007, not significant (p=0.20) — expected null per the plan's own priors; no R7 queued, Phase 2 stays closed. Full per-round detail in `experiments/2026-06-{15,16,17,18,19}-*.md`, `experiments/2026-07-07-user-lora-lamp4-round6-multi.md`, and memory `project_user_lora_lamp4_single_user_retrospective.md`, `project_user_lora_round5_lamp3_design.md`, `project_user_lora_round6_lamp4_design.md`.

**R6 carryovers from R5 (settled, not relitigated):** OPPU recipe verbatim (r=8, q+v only, alpha=16, dropout=0.05, AdamW, LR=1e-5, L2=1e-2, cosine + 3% warmup, 3 epochs, save_strategy=epoch, save_total_limit=1); per_device=2 / grad_accum=4 (R5's final working config — skip the OOM iteration); base = SmolLM3-3B + A1-lamp ckpt-1000 stacked via `--base-adapter`; eval = BM25 k=4 + greedy + seed=0 + `enable_thinking=False` + max_new_tokens=64; smoke = one user (smallest profile_size).

---

## Model & training

- **Base:** `HuggingFaceTB/SmolLM3-3B`, bf16, frozen.
- **Framework:** HuggingFace Transformers + PEFT.
- **Loss:** CE only. **No KD, no teacher co-loading, no base-weight modification.** Teacher is offline data generation only (and that whole branch is dropped — see hard constraints).

**Task-LoRA config:**
```python
LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```
r=4 (vs original spec r=64) because the value prop is on-device efficiency — adapter params scale linearly in r; alpha drops proportionally (alpha/r=2). See OPPU arXiv:2402.04401 for typical r=4–16 mobile configs.

**Training setup:** AdamW lr=3e-4, cosine + 3% warmup, per-device bs=4 × grad_accum=8 (effective 32), 2–3 epochs, checkpoint every 500 steps. Metrics streamed to `metrics.jsonl` + `train_meta.json` summary. W&B wired up in `train.py` but defaults off (flip `report_to` in config to re-enable).

---

## Datasets

- **LaMP user-based split** at `data/lamp/LaMP_{3,4,7}/` — used for A1-lamp Task-LoRA training (users disjoint across train/dev/test).
- **LaMP time-based split** at `data/lamp_time/LaMP_{3,4,7}/` — same users in every split, partitioned chronologically. Used for User-LoRA. Downloaded 2026-06-10 via `data/download_lamp.py --split-type time`. Test_outputs present (not withheld); profile entries carry `date` for the partition.
- **Per-user volume varies sharply by task** (`experiments/2026-06-12-lamp-time-split-per-user-counts.md`): LaMP_4 ~7.5 records/user avg with records framing; LaMP_3 richest with profile-entry reframing (~175 review→rating pairs/user); LaMP_7 stuck at 1–2 examples/user regardless. Tasks: LaMP-3 (rating prediction), LaMP-4 (headline gen), LaMP-7 (tweet paraphrase). **LaMP-6 unsupported** — only Avocado email file-id placeholders ship; needs licensed corpus.
- **Built corpora:** `data/lamp_train_{LaMP_3,LaMP_4,LaMP_7,mixed}_bm25k4.jsonl` (42,964 examples in `mixed`; 20,000 + 12,527 + 10,437; 87.9 MB), provenance in `lamp_train_mixed_bm25k4.meta.json`.
- **Synthetic preference-conditional data and domain-specific A2 corpora are DROPPED** (2026-06-02 pivot — Q2/Q3 dropped).

---

## Evaluation

| Task | Metric |
|---|---|
| LaMP-3 (rating prediction) | Accuracy |
| LaMP-4 (headline gen) | ROUGE-1 |
| LaMP-7 (tweet paraphrase) | ROUGE-1 |

Plus **BFCL AST regression** before/after each Task-LoRA training run (target ≥90; baseline 92.3; sanity check only).

**Comparison chain (post-pivot):** Baseline → A1-lamp (Q1, answered) → A1-lamp + User-LoRA (Q4, answered YES for LaMP-3; answered NULL for LaMP-4 per R6).

---

## Repo structure

```
/
├── CLAUDE.md, Dockerfile, requirements.txt, pyrightconfig.json
├── condor/                       # Condor submit files + helper scripts
│   ├── build_dataset.sub         # CPU: preprocess LaMP train → JSONL
│   ├── download_model.{py,sub}   # one-time HF Hub pull of SmolLM3-3B
│   ├── download_llama.{py,sub}   # one-time HF Hub pull of Llama-3.1-{8B,70B}-Instruct
│   ├── interactive.sub           # CPU shell for ad-hoc inspection (uncomment GPU block for smoke tests)
│   ├── eval_lamp.sub             # LaMP profile-baseline + adapter eval (×3 parallel)
│   ├── eval_lamp_floor.sub       # LaMP non-personalized floor (×3 parallel)
│   ├── eval_lamp_llama.sub       # Llama scale comparator, full test sets (12 jobs)
│   ├── eval_lamp_llama_k100.sub  # Llama scale comparator, K=100 subset (4 jobs)
│   ├── eval_bfcl.sub             # BFCL AST regression (1 GPU, all categories)
│   ├── train.sub                 # superseded (2-epoch A1-lamp)
│   ├── train_1ep.sub             # canonical (1-epoch A1-lamp)
│   ├── chat.py                   # REPL with model + optional adapter
│   └── smoke_test.py             # Docker-image env check
├── data/
│   ├── download_lamp.py          # `--split-type {user,time}` (default user)
│   ├── lamp/                     # user-based split — A1-lamp training
│   ├── lamp_time/                # time-based split — User-LoRA
│   ├── lamp_user_stats.py        # per-user record-count analysis
│   ├── lamp_user_stats/          # per-task user CSVs + R5/R6 top-K JSONs
│   ├── models/SmolLM3-3B/        # downloaded weights (~6 GB)
│   ├── models/Llama-3.1-{8B,70B}-Instruct → /scratch/<group>/<user>/models/  # symlinks; ~16 + ~141 GB on /scratch
│   ├── lamp_train_*_bm25k4.jsonl # built by build_dataset.py
│   └── lamp_train_mixed_bm25k4.meta.json
├── train/
│   ├── build_dataset.py          # raw LaMP train → BM25-retrieved JSONL
│   ├── build_user_dataset.py     # per-user variant (User-LoRA)
│   ├── train.py                  # SFT trainer — config-driven, SmolLM3 chat template (thinking off), loss-masked to assistant, supports --base-adapter
│   ├── config/
│   │   ├── a1_lamp.json          # superseded (2-epoch)
│   │   ├── a1_lamp_1ep.json      # canonical (1-epoch, → checkpoint-1000)
│   │   └── user_lora_*.json      # R5/R6 OPPU templates
│   └── checkpoints/              # training output (gitignored)
├── eval/
│   ├── eval_lamp.py              # LaMP harness (BM25 k=4, refuse-to-overwrite, --base-adapter, --user-records, --user-records-from-file, --resume, --device-map)
│   ├── eval_bfcl.py              # BFCL via bfcl-eval's ast_checker as a library
│   ├── paired_compare.py         # single-user paired stats (User-LoRA R1-R4)
│   ├── paired_compare_per_user.py # multi-user grouped paired stats (R5/R6)
│   ├── bench_aggregate.py        # on-device bench JSONL → aggregate JSON
│   ├── tables.py                 # Llama-scale Tables 1 + 2 from results/*.json (markdown or plain)
│   └── summary.py                # flatten results/*.json → table
├── ios/mlx-swift-examples/       # git-subtree vendored (upstream 378f244); LLMEval edited for SmolLM3 + benchmark harness
├── results/                      # flat scalar JSON + per-example predictions JSONL; ondevice/ subdir for bench telemetry
├── runlogs/                      # Condor stdout/stderr (gitignored)
├── experiments/                  # YYYY-MM-DD-<slug>.md per run
└── notebooks/                    # personal analysis (gitignored)
```

---

## Standard script patterns (converged across all eval/train/data-prep scripts)

- **Provenance banner at startup** — first stdout line prints task / split / condition / seed / commit short SHA / Condor cluster.proc IDs / host.
- **Provenance dict in every result record** — `git_commit`, `git_dirty`, `condor_cluster_id`, `condor_proc_id`, `hostname`, `timestamp_utc`, library versions.
- **Flat single-level JSON result records** — every field a scalar, so `pd.DataFrame([json.load(open(p)) for p in glob("results/*.json")])` works with zero unnesting.
- **Per-example predictions in a sibling JSONL** — one `{id, pred, gold}` per line (BFCL adds `category`, `pred_text`, `pred_parsed`, `valid`, `error_type`).
- **Refuse-to-overwrite by default** — every output-producing script checks existing files and `sys.exit(1)` unless `--overwrite` is passed. Smoke runs (`--limit > 0`) get an `_limitN` filename suffix so they can't collide with full-run outputs even if `--overwrite` was used.
- **Condor IDs forwarded via submit file's `environment`**: `CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)` — script reads via `os.environ.get`. Links result records back to runlog files.

---

## Eval methodology choices (frozen — don't relitigate)

- **LaMP personalization channel = BM25 top-k retrieval** (k=4) of the user's profile into the `system` slot. Not summarization (tried, reverted — see `notebooks/lamp_evaluation_approach.md`). Same BM25, same k, same per-task formatting, same role layout at training time (`build_dataset.py`) and eval time (`eval_lamp.py`). Train/eval consistency is the cardinal rule.
- **System-always prompt regime** (resolved 2026-05-31) — BM25 profile sits in `system` for every training example, so the Task-LoRA expects that shape at inference. Open hypothesis is on-device User-LoRA could absorb the profile into adapter weights and drop the `system` prompt tax (+118 to +482 tokens/query).
- **BFCL eval uses Path C** — install bfcl-eval in the image, generate via our own transformers stack, call `ast_checker` as a library on outputs. SmolLM3 isn't in BFCL's `MODEL_CONFIG_MAPPING`, so we pass `model_name="meta-llama/Llama-3.1-8B-Instruct"` as a neutral placeholder (recorded as `scorer_model_name_placeholder`); `BFCL_PROJECT_ROOT` must be set before any `bfcl_eval` import (`eval_bfcl.py` sets it to `/tmp/bfcl_project_root`).
- **BFCL `irrelevance` skipped** (data file `possible_answer/BFCL_v4_irrelevance.json` doesn't ship — correct answer is "no call"). Could be extended in ~10 lines to score `correct iff pred_parsed == []`.
- **BFCL Java/JS errors not investigated** — 80 `type_error:{java,js}` account for most of the 80.78 vs 92.3 gap. Worth a 2-min spot-check before post-training comparison.

---

## Docker image

Current tag: **`ghcr.io/gordofreemo/smollm3-train:ver4`**.
1. Base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11, torch 2.5.1+cu124)
2. `apt-get install git` (added ver3) — for `git_commit` provenance inside the container
3. `pip install -r requirements.txt` — transformers, peft, datasets, accelerate, wandb, rouge_score, bfcl-eval, soundfile

**When you change `requirements.txt` or the Dockerfile**, bump the tag and update **all ten** sub files: `condor/{eval_lamp,eval_lamp_floor,eval_lamp_llama,eval_lamp_llama_k100,eval_bfcl,build_dataset,train,interactive,download_model,download_llama}.sub`.

**GPU capability ceiling.** The cluster has RTX PRO 6000 Blackwell (sm_120) nodes that ver4's PyTorch 2.5.1+cu124 cannot target. Llama submits constrain to `Capability >= 8.0 && Capability < 10.0` via `require_gpus` — anything that lands on Blackwell dies at the first CUDA op with "no kernel image is available for execution on the device". Other submits don't yet carry this guard; harden them if they start failing the same way.

---

## Hard constraints

- **Never modify base model weights.** LoRA only; base frozen.
- **No KD loss.** CE only.
- **No co-loading teacher and student.** (Teacher branch dropped entirely with the 2026-06-02 pivot.)
- **Reproducibility first.** Every training run launchable from one CLI command with fixed seed. Log full command in the experiment file.
- **No profile leakage between splits.** Validate explicitly. For time-based splits this means no overlap within a user between train-period and dev-period interactions — enforced by LaMP's split by construction.
- **No on-device / mobile code** — **LIFTED for Phase 3** only. Phases 1 & 2 retain it as historical framing.

---

## Experiment log format

Every run gets `experiments/YYYY-MM-DD-<slug>.md`:

```markdown
## Hypothesis
## Setup (command, config, seed)
## Result (loss curve, eval numbers)
## Conclusion
```

---

## Key references (do not hallucinate URLs)

- SmolLM3-3B: `HuggingFaceTB/SmolLM3-3B` on HuggingFace
- LaMP benchmark: lamp-benchmark.github.io
- OPPU (per-user PEFT recipe used in R5/R6): arXiv 2402.04401
- BFCL: gorilla.cs.berkeley.edu/leaderboard.html
- CDCDA-PLM (closest prior work — cloud synthetic + on-device PEFT + LaMP): arXiv 2508.21313
- Apple on-device fine-tuning (memory-efficient backprop): arXiv 2510.03425
