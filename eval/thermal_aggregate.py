#!/usr/bin/env python3
"""Aggregate h10 thermal-cooldown telemetry → recovery curve + duty-cycle verdict.

Reads one or more `train_bench_metrics_thermal.jsonl` pulls, groups by
`bench_session_id`, and for each session computes:

  * the session's own COLD reference (pre-soak probe) and HOT steady state
    (soak plateau) — their ratio R is the in-session cold/hot speedup, which
    is why this round needs no constant borrowed from h7;
  * the recovery curve r(t) over the observation window, and the
    t50/t80/t90/t95 milestones;
  * the enum-vs-reality gap: when `thermalState` first said `nominal` versus
    when throughput actually recovered;
  * the pre-registered duty-cycle verdict, under a bound written in
    duty-cycling's favour.

Duty-cycle verdict — TWO estimators, and the difference between them matters.

(1) SOAK-ANCHORED (primary). The soak IS a measurement of how much work a
burst starting from a cold device actually produces: `soak_iterations`
iterations in `soak_span_s` seconds, re-heating included. So

    duty_rate(T)  = soak_iterations / (soak_span_s + T)
    ratio_soak(T) = duty_rate(T) / (1 / hot_spi)

This is still an upper bound on duty-cycling, for a reason worth stating:
after only T seconds of cooling the die is cool but the chassis is not (the
enum is still `serious` long after throughput recovers), so a real repeat
burst would re-heat FASTER than the cold first soak did and yield fewer than
`soak_iterations`. Granting the full cold-start work output is therefore
generous, and a verdict of "continuous wins" under it is airtight.

(2) INSTANTANEOUS-PROBE (secondary, diagnostic only, NOT the verdict).
    ratio_probe(T) = [rate(T) / hot_rate] * B / (B + T)
i.e. assume a burst holds its starting rate for the whole burst. This is the
bound the plan doc pre-registered, and h10's own data shows it is NOT tight:
recovery takes ~6 min while re-heating takes ~1-2 min, so the assumption is
violated almost immediately and the estimator inflates badly (Run A: 1.94x
vs. the soak-anchored 1.03x). Kept because it is what was pre-registered and
because the GAP between the two estimators is itself the finding — it is
precisely the cost of ignoring re-throttling. The plan doc's `B/(B+T) > 1/R`
is the further special case where recovery is complete.

Both are reported, and `duty_cycle_verdict` follows the SOAK-ANCHORED one.

(3) MEASURED (the sustained-cycling arm) — and this one supersedes both.
Estimators (1) and (2) extrapolate a repeating schedule from a SINGLE burst,
and the first burst is the only one that starts from a genuinely cool chassis,
so both are upper bounds. A `--benchmark-thermal-cycle` session runs the
schedule for real and is summarised by `summarize_cycle_session`, reporting
iterations-per-burst across cycles plus a sustained rate. Where such a session
is present it is the answer; the aggregator prints an explicit note when a
single-burst session claims a win that the measured arm contradicts. On this
hardware they disagree sharply — 1.246x extrapolated vs. 0.755x measured.

Stdlib only, matching this repo's other aggregators.

See experiments/2026-07-28-ondevice-thermal-cooldown-h10-plan.md.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone

# Recovery milestones reported for every session (fraction of the
# soak-induced throughput gap that has closed).
MILESTONES = (0.50, 0.80, 0.90, 0.95)

# Soak-plateau window: mean over the LAST this-many seconds of the soak. The
# h5 E2E data shows throughput flattening by ~40-50 min, so the tail of a
# 60-min soak is genuinely in steady state; a short soak (Run C) may never
# reach one, which `plateau_is_steady_state` flags rather than hides.
PLATEAU_TAIL_SECONDS = 600.0

# Below this soak duration the plateau estimate is not trustworthy as a HOT
# steady state (Run C's 10-minute soak is deliberately in this regime).
STEADY_STATE_MIN_SOAK_SECONDS = 2400.0

# Duty-cycle verdicts inside +/- this band are reported as a wash. Set from
# the observed probe-to-probe scatter on Run A (~1.7% across adjacent probes,
# each contributing one kept iteration).
DUTY_TOLERANCE = 0.02

# A self-limit phase shorter than this cannot be trusted as an equilibrium.
# Learned the hard way: the h10d pilot used 20-min phases, and its D=0 phase
# (which IS continuous training) read 8.05 s/iter instead of the true 9.98,
# because the device needs 40-50 min to plateau. Phase means from short phases
# are ramps, not steady states.
SELFLIMIT_MIN_PHASE_SECONDS = 2400.0


def git_provenance():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    commit = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return commit, bool(dirty) if dirty != "unknown" else False


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write can leave a torn final line.
                    continue
    return rows


def kept_iters(records):
    """Probe/cold-ref iterations after discarding index 0.

    Iteration 0 carries MLX graph-compile / first-allocation overhead plus the
    forced iteration-0 validation pass folded into the same window — the same
    discard h7 applies. The device logs every iteration; the rule lives here.
    """
    return [r for r in records if r.get("iter_index", 0) >= 1]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def interpolate_crossing(points, threshold):
    """First t where recovery >= threshold, linearly interpolated.

    `points` is [(t, recovery)] in ascending t. Returns None if the curve
    never reaches the threshold within the observed window — a real result
    for this round, not a failure.
    """
    prev_t = prev_r = None
    for t, r in points:
        if r >= threshold:
            if prev_t is None or r == prev_r:
                return t
            span = r - prev_r
            if span <= 0:
                return t
            return prev_t + (threshold - prev_r) * (t - prev_t) / span
        prev_t, prev_r = t, r
    return None


def summarize_cycle_session(sid, rows, by_type):
    """Sustained-cycling arm (h10c): does iterations-per-burst hold up?

    Runs A/B/C measure one burst plus the recovery after it, and any burst
    schedule extrapolated from that is an upper bound — the first burst is the
    only one starting from a genuinely cool chassis. This arm runs the schedule
    for real, so the decisive output is the per-burst iteration count across
    cycles.

    The continuous-training baseline cannot come from this session (it never
    trains continuously), so it is borrowed from the long-soak cooldown
    sessions in `apply_reference_plateau`. Two rates are reported:

      * sustained  = iterations / wall-clock over cycles 2..N, i.e. what the
        schedule actually delivers, rest gaps included;
      * in-burst   = iterations / time spent training, i.e. the same schedule
        with all restart cost forgiven. Reported so restart overhead can be
        ruled in or out as the explanation rather than assumed either way.

    Cycle 1 is excluded from the sustained figure: it is the cold-start burst,
    and including it would re-introduce exactly the bias this arm exists to
    measure.
    """
    start = next(iter(by_type.get("cycle_run_start", [])), None)
    end = next(iter(by_type.get("cycle_run_end", [])), None)
    meta_src = start or (rows[0] if rows else {})

    cold_iters = kept_iters(by_type.get("cold_ref", []))
    cold_spi = mean([r["seconds_per_iter"] for r in cold_iters])

    ends = {r["cycle_index"]: r for r in by_type.get("cycle_burst_end", [])}
    bursts = {}
    for r in by_type.get("cycle_burst", []):
        bursts.setdefault(r["probe_index"], []).append(r)

    per_cycle = []
    for k in sorted(bursts):
        its = sorted(bursts[k], key=lambda r: r["elapsed_s"])
        train_s = sum(r["seconds_per_iter"] for r in its)
        dur = ends[k]["burst_duration_s"] if k in ends else None
        per_cycle.append(
            {
                "cycle_index": k,
                "iterations": len(its),
                "train_seconds": train_s,
                "burst_duration_s": dur,
                # Wall time inside the burst not spent in a training iteration
                # (LoRA re-apply, tokenisation, graph compile).
                "setup_overhead_s": (dur - train_s) if dur is not None else None,
                "seconds_per_iter": train_s / len(its) if its else None,
                "thermal_state": ends[k]["thermal_state"] if k in ends else None,
            }
        )

    rest = []
    for r in by_type.get("cycle_rest_probe", []):
        if r.get("iter_index", 0) >= 1:
            rest.append(
                {
                    "after_cycle": r["probe_index"],
                    "seconds_per_iter": r["seconds_per_iter"],
                    "rest_elapsed_s": r.get("cooldown_elapsed_s"),
                }
            )
    rest.sort(key=lambda x: x["after_cycle"])

    settled = [c for c in per_cycle if c["cycle_index"] >= 1]
    sustained_iters = sum(c["iterations"] for c in settled)
    sustained_span = None
    if settled and 0 in ends and settled[-1]["cycle_index"] in ends:
        sustained_span = (
            ends[settled[-1]["cycle_index"]]["elapsed_s"] - ends[0]["elapsed_s"]
        )
    sustained_rate = (
        sustained_iters / sustained_span if (sustained_span and sustained_iters) else None
    )
    inburst_train = sum(c["train_seconds"] for c in settled)
    inburst_rate = sustained_iters / inburst_train if inburst_train else None

    return {
        "bench_session_id": sid,
        "session_type": "cycle",
        "app_build": meta_src.get("app_build"),
        "bench_schema_version": meta_src.get("bench_schema_version"),
        "model": meta_src.get("model"),
        "device_model": meta_src.get("device_model"),
        "num_lora_layers": meta_src.get("num_lora_layers"),
        "timestamp_utc": meta_src.get("timestamp_utc"),
        "complete": end is not None,
        "run_elapsed_s": end.get("elapsed_s") if end else None,
        "burst_seconds": start.get("burst_seconds") if start else None,
        "rest_seconds": start.get("rest_seconds") if start else None,
        "cycles_planned": start.get("cycles_planned") if start else None,
        "cycles_observed": len(per_cycle),
        "cold_ref_seconds_per_iter": cold_spi,
        "battery_level_start": start.get("battery_level") if start else None,
        "battery_level_end": end.get("battery_level_end") if end else None,
        "first_burst_iterations": per_cycle[0]["iterations"] if per_cycle else None,
        "settled_burst_iterations_mean": mean([c["iterations"] for c in settled]),
        "settled_seconds_per_iter": mean([c["seconds_per_iter"] for c in settled]),
        "setup_overhead_s_mean": mean(
            [c["setup_overhead_s"] for c in per_cycle if c["setup_overhead_s"] is not None]
        ),
        "sustained_iterations": sustained_iters,
        "sustained_span_s": sustained_span,
        "sustained_iter_per_s": sustained_rate,
        "inburst_iter_per_s": inburst_rate,
        # Filled in by apply_reference_plateau (needs the cooldown sessions).
        "reference_plateau_seconds_per_iter": None,
        "reference_plateau_source": None,
        "sustained_vs_continuous": None,
        "inburst_vs_continuous": None,
        "duty_cycle_verdict": None,
        "_cycles": per_cycle,
        "_rest_probes": rest,
    }


def summarize_selflimit_session(sid, rows, by_type):
    """Self-limiting arm (h10d): pace the loop, don't stop it.

    A fixed delay D is inserted after every iteration (finest granularity the
    API allows), so duty u = c/(c+D) where c is real compute time. Effective
    time per iteration is c + D, and the arm wins iff c + D < the continuous
    equilibrium.

    Reported per phase: tail-mean compute, effective time, duty, ratio vs the
    continuous reference, and — critically — `converged`, because a phase
    shorter than SELFLIMIT_MIN_PHASE_SECONDS is a heat-up ramp and its mean is
    NOT an equilibrium. The exchange rate dc/dD across phases is the quantity
    that decides the whole axis: pacing needs steeper than -1.
    """
    start = next(iter(by_type.get("selflimit_run_start", [])), None)
    end = next(iter(by_type.get("selflimit_run_end", [])), None)
    meta_src = start or (rows[0] if rows else {})

    cold_iters = kept_iters(by_type.get("cold_ref", []))
    cold_spi = mean([r["seconds_per_iter"] for r in cold_iters])

    phases = {}
    for r in by_type.get("selflimit_train", []):
        phases.setdefault(r["phase_index"], []).append(r)

    per_phase = []
    for k in sorted(phases):
        its = sorted(phases[k], key=lambda r: r["phase_elapsed_s"])
        D = its[0]["delay_s"]
        span = its[-1]["phase_elapsed_s"]
        tail = [r for r in its if r["phase_elapsed_s"] >= span - PLATEAU_TAIL_SECONDS]
        c = mean([r["seconds_per_iter"] for r in tail])
        # Is it still climbing? Compare the first and last thirds of the tail
        # window; a rising trend means the phase never settled.
        third = max(1, len(tail) // 3)
        drift = (
            mean([r["seconds_per_iter"] for r in tail[-third:]])
            - mean([r["seconds_per_iter"] for r in tail[:third]])
        )
        per_phase.append(
            {
                "phase_index": k,
                "delay_s": D,
                "iterations": len(its),
                "phase_span_s": span,
                "compute_seconds_per_iter": c,
                "effective_seconds_per_iter": c + D,
                "duty": c / (c + D) if (c + D) else None,
                "drift_s_per_iter": drift,
                "converged": bool(span >= SELFLIMIT_MIN_PHASE_SECONDS and abs(drift) < 0.15),
                "thermal_state": its[-1].get("thermal_state"),
            }
        )

    return {
        "bench_session_id": sid,
        "session_type": "selflimit",
        "app_build": meta_src.get("app_build"),
        "model": meta_src.get("model"),
        "device_model": meta_src.get("device_model"),
        "num_lora_layers": meta_src.get("num_lora_layers"),
        "timestamp_utc": meta_src.get("timestamp_utc"),
        "complete": end is not None,
        "run_elapsed_s": end.get("elapsed_s") if end else None,
        "total_planned_s": meta_src.get("total_planned_s"),
        "cold_ref_seconds_per_iter": cold_spi,
        "battery_level_start": start.get("battery_level") if start else None,
        "battery_level_end": end.get("battery_level_end") if end else None,
        "n_phases": len(per_phase),
        # Filled by apply_reference_plateau.
        "reference_plateau_seconds_per_iter": None,
        "reference_plateau_source": None,
        "best_effective_seconds_per_iter": None,
        "best_delay_s": None,
        "best_vs_continuous": None,
        "exchange_rate_dc_dD": None,
        "duty_cycle_verdict": None,
        "_phases": per_phase,
    }


def summarize_session(sid, rows):
    by_type = {}
    for r in rows:
        by_type.setdefault(r["record_type"], []).append(r)

    # The sustained-cycling arm has an entirely different record shape (no
    # soak, no observation window) and is summarised separately.
    if by_type.get("cycle_run_start") or by_type.get("cycle_burst"):
        return summarize_cycle_session(sid, rows, by_type)
    if by_type.get("selflimit_run_start") or by_type.get("selflimit_train"):
        return summarize_selflimit_session(sid, rows, by_type)

    start = next(iter(by_type.get("run_start", [])), None)
    soak_end_rec = next(iter(by_type.get("soak_end", [])), None)
    end = next(iter(by_type.get("run_end", [])), None)

    meta_src = start or (rows[0] if rows else {})
    soak_seconds = meta_src.get("soak_seconds")
    probe_interval = meta_src.get("probe_interval_s")

    # --- Cold reference: the session's own 100%-recovered baseline. ---
    cold_iters = kept_iters(by_type.get("cold_ref", []))
    cold_spi = mean([r["seconds_per_iter"] for r in cold_iters])

    # --- Hot steady state: tail of the soak. ---
    soak = sorted(by_type.get("soak", []), key=lambda r: r["elapsed_s"])
    hot_spi = None
    plateau_n = 0
    if soak:
        soak_last = soak[-1]["elapsed_s"]
        tail = [r for r in soak if r["elapsed_s"] >= soak_last - PLATEAU_TAIL_SECONDS]
        plateau_n = len(tail)
        hot_spi = mean([r["seconds_per_iter"] for r in tail])

    ratio_R = (hot_spi / cold_spi) if (hot_spi and cold_spi) else None

    # --- Recovery curve over the observation window. ---
    probes = {}
    for r in by_type.get("probe", []):
        probes.setdefault(r["probe_index"], []).append(r)

    curve = []
    for idx in sorted(probes):
        iters = kept_iters(probes[idx])
        spi = mean([r["seconds_per_iter"] for r in iters])
        if spi is None:
            continue
        t = probes[idx][0].get("cooldown_elapsed_s")
        rec = None
        if cold_spi and hot_spi and hot_spi != cold_spi:
            # Fraction of the soak-induced gap that has closed.
            rec = (hot_spi - spi) / (hot_spi - cold_spi)
        curve.append(
            {
                "probe_index": idx,
                "cooldown_elapsed_s": t,
                "seconds_per_iter": spi,
                "recovery": rec,
                "thermal_state": probes[idx][0].get("thermal_state"),
                "n_iters_kept": len(iters),
            }
        )

    pts = [(c["cooldown_elapsed_s"], c["recovery"]) for c in curve if c["recovery"] is not None]
    milestones = {
        f"t{int(m * 100)}_s": interpolate_crossing(pts, m) for m in MILESTONES
    }

    # --- Enum vs reality. ---
    soak_end_elapsed = soak_end_rec.get("elapsed_s") if soak_end_rec else None
    enum_nominal_s = None
    if soak_end_elapsed is not None:
        post = sorted(
            (
                r
                for r in by_type.get("sample", [])
                if r.get("elapsed_s", -1) >= soak_end_elapsed
            ),
            key=lambda r: r["elapsed_s"],
        )
        for r in post:
            if r.get("thermal_state") == "nominal":
                enum_nominal_s = r["elapsed_s"] - soak_end_elapsed
                break

    t90 = milestones.get("t90_s")
    enum_gap_s = None
    if enum_nominal_s is not None and t90 is not None:
        # Positive => throughput lagged the enum, i.e. the enum declared the
        # device cool while it was still measurably throttled.
        enum_gap_s = t90 - enum_nominal_s

    # --- Work actually produced by one cold-start burst (the soak). ---
    soak_iterations = len(soak)
    soak_span_s = None
    if soak and soak_end_rec is not None:
        # Time to produce those iterations: back off the first record's own
        # iteration time, since its elapsed_s is stamped at completion.
        soak_span_s = (
            soak_end_rec["elapsed_s"] - soak[0]["elapsed_s"] + soak[0]["seconds_per_iter"]
        )

    # --- Duty-cycle verdict (two estimators; see module docstring). ---
    duty = []
    best = None
    best_probe = None
    if hot_spi and soak_seconds:
        B = soak_seconds
        continuous_rate = 1.0 / hot_spi
        for c in curve:
            T = c["cooldown_elapsed_s"]
            if T is None or c["seconds_per_iter"] is None:
                continue
            # (2) instantaneous-probe bound — diagnostic only.
            ratio_probe = (hot_spi / c["seconds_per_iter"]) * B / (B + T)
            # (1) soak-anchored — the verdict. A burst starting at recovery
            # level r(T) produces work between the steady-state amount (r=0,
            # i.e. no cooling) and the measured cold-start amount (r=1). This
            # anchoring is what makes ratio(0) == 1 exactly: with no cooling
            # you ARE running continuously.
            ratio_soak = None
            if soak_span_s and soak_iterations and c["recovery"] is not None:
                steady_work = soak_span_s / hot_spi
                r = max(0.0, min(1.0, c["recovery"]))
                work = steady_work + r * (soak_iterations - steady_work)
                duty_rate = work / (soak_span_s + T)
                ratio_soak = duty_rate / continuous_rate
            row = {
                "cooldown_s": T,
                "throughput_vs_continuous": ratio_soak,
                "throughput_vs_continuous_probe_bound": ratio_probe,
            }
            duty.append(row)
            if ratio_soak is not None and (
                best is None or ratio_soak > best["throughput_vs_continuous"]
            ):
                best = row
            if best_probe is None or ratio_probe > best_probe[1]:
                best_probe = (T, ratio_probe)

    charging = [r for r in by_type.get("sample", []) if r.get("charging") is not None]
    stayed_charging = all(r["charging"] for r in charging) if charging else None

    return {
        "bench_session_id": sid,
        "app_build": meta_src.get("app_build"),
        "bench_schema_version": meta_src.get("bench_schema_version"),
        "model": meta_src.get("model"),
        "device_model": meta_src.get("device_model"),
        "num_lora_layers": meta_src.get("num_lora_layers"),
        "soak_seconds": soak_seconds,
        "probe_interval_s": probe_interval,
        "probe_tokens": meta_src.get("probe_tokens"),
        "probe_iterations": meta_src.get("probe_iterations"),
        "timestamp_utc": meta_src.get("timestamp_utc"),
        "complete": end is not None,
        "run_elapsed_s": end.get("elapsed_s") if end else None,
        "battery_level_start": start.get("battery_level") if start else None,
        "battery_level_end": end.get("battery_level_end") if end else None,
        "stayed_charging": stayed_charging,
        "cold_ref_seconds_per_iter": cold_spi,
        "cold_ref_n_iters": len(cold_iters),
        "soak_plateau_seconds_per_iter": hot_spi,
        "soak_plateau_n_iters": plateau_n,
        "soak_n_iters": len(soak),
        # True only when the soak was long enough to plausibly reach steady
        # state (Run C's short soak deliberately is not).
        "plateau_is_steady_state": (
            bool(soak_seconds and soak_seconds >= STEADY_STATE_MIN_SOAK_SECONDS)
        ),
        "cold_hot_ratio_R": ratio_R,
        "n_probes": len(curve),
        **milestones,
        "enum_first_nominal_s": enum_nominal_s,
        "enum_vs_t90_gap_s": enum_gap_s,
        "soak_iterations": soak_iterations,
        "soak_span_s": soak_span_s,
        # Structural ceiling on ANY burst-and-cool schedule at this burst
        # length: the cold-start bonus, i.e. how much more work a burst
        # starting cold produces than one at steady state. Only reachable with
        # instantaneous cooling, so the real best is strictly below it.
        "max_possible_gain": (
            (soak_iterations / (soak_span_s / hot_spi) - 1.0)
            if (soak_span_s and hot_spi and soak_iterations)
            else None
        ),
        "best_cooldown_s": best["cooldown_s"] if best else None,
        "best_throughput_vs_continuous": (
            best["throughput_vs_continuous"] if best else None
        ),
        # Three-way, not two: adjacent probes scatter ~+/-1.7% (n=1 kept
        # iteration each), so a "win" inside that band is noise, not a
        # schedule worth adopting.
        "duty_cycle_verdict": (
            None
            if best is None
            else (
                "duty_cycling_wins"
                if best["throughput_vs_continuous"] > 1.0 + DUTY_TOLERANCE
                else (
                    "continuous_wins"
                    if best["throughput_vs_continuous"] < 1.0 - DUTY_TOLERANCE
                    else "wash"
                )
            )
        ),
        "best_cooldown_s_probe_bound": best_probe[0] if best_probe else None,
        "best_throughput_vs_continuous_probe_bound": (
            best_probe[1] if best_probe else None
        ),
        "_curve": curve,
        "_duty": duty,
    }


def apply_reference_plateau(summaries):
    """Recompute the duty-cycle verdict for short-soak sessions.

    A short soak never reaches the throttled steady state that continuous
    training actually sits in (a 10-min burst ends around 6.6-7.6 s/iter vs.
    the ~10.0 s/iter a 60-min burst settles at). Using such a session's own
    plateau as the "continuous training" reference understates continuous cost
    badly and makes the ratio meaningless — it reports ~1.00x by construction.

    So: sessions whose soak WAS long enough contribute a reference plateau
    (mean across them), and short-soak sessions are re-scored against it. The
    burst work and recovery curve are the short session's own; only the
    continuous-training baseline is borrowed. `reference_plateau_source`
    records which was used.
    """
    steady = [
        s["soak_plateau_seconds_per_iter"]
        for s in summaries
        if s.get("session_type") not in ("cycle", "selflimit")
        and s["plateau_is_steady_state"]
        and s["soak_plateau_seconds_per_iter"]
    ]
    ref = mean(steady)
    src = f"long_soak_mean_n{len(steady)}"

    # Self-limit sessions: the D=0 phase, if long enough, IS continuous
    # training and can serve as its own reference; otherwise borrow. Then the
    # verdict is min(effective) vs that reference, and the exchange rate
    # dc/dD is the chord from D=0 to the largest delay measured.
    for s in summaries:
        if s.get("session_type") != "selflimit":
            continue
        own = [
            p for p in s["_phases"]
            if p["delay_s"] == 0 and p["converged"] and p["compute_seconds_per_iter"]
        ]
        if own:
            s["reference_plateau_seconds_per_iter"] = mean(
                [p["compute_seconds_per_iter"] for p in own])
            s["reference_plateau_source"] = "own_D0_phase"
        else:
            s["reference_plateau_seconds_per_iter"] = ref
            s["reference_plateau_source"] = src if ref else None
        r = s["reference_plateau_seconds_per_iter"]
        if not r:
            continue
        paced = [p for p in s["_phases"] if p["delay_s"] > 0]
        if paced:
            best = min(paced, key=lambda p: p["effective_seconds_per_iter"])
            s["best_effective_seconds_per_iter"] = best["effective_seconds_per_iter"]
            s["best_delay_s"] = best["delay_s"]
            s["best_vs_continuous"] = r / best["effective_seconds_per_iter"]
            v = s["best_vs_continuous"]
            s["duty_cycle_verdict"] = (
                "duty_cycling_wins" if v > 1.0 + DUTY_TOLERANCE
                else ("continuous_wins" if v < 1.0 - DUTY_TOLERANCE else "wash")
            )
            # Chord from the continuous reference to the largest measured delay.
            # Pacing can only win if this is steeper than -1.
            far = max(paced, key=lambda p: p["delay_s"])
            s["exchange_rate_dc_dD"] = (
                (far["compute_seconds_per_iter"] - r) / far["delay_s"]
            )

    # Cycle sessions never train continuously, so their continuous-training
    # baseline has to be borrowed outright.
    for s in summaries:
        if s.get("session_type") != "cycle":
            continue
        s["reference_plateau_seconds_per_iter"] = ref
        s["reference_plateau_source"] = src if ref else None
        if not ref:
            continue
        continuous_rate = 1.0 / ref
        if s["sustained_iter_per_s"]:
            s["sustained_vs_continuous"] = s["sustained_iter_per_s"] / continuous_rate
        if s["inburst_iter_per_s"]:
            s["inburst_vs_continuous"] = s["inburst_iter_per_s"] / continuous_rate
        v = s["sustained_vs_continuous"]
        if v is not None:
            s["duty_cycle_verdict"] = (
                "duty_cycling_wins"
                if v > 1.0 + DUTY_TOLERANCE
                else ("continuous_wins" if v < 1.0 - DUTY_TOLERANCE else "wash")
            )

    for s in summaries:
        if s.get("session_type") in ("cycle", "selflimit"):
            continue
        if s["plateau_is_steady_state"] or not ref:
            s["reference_plateau_seconds_per_iter"] = s["soak_plateau_seconds_per_iter"]
            s["reference_plateau_source"] = "own_soak"
            continue
        s["reference_plateau_seconds_per_iter"] = ref
        s["reference_plateau_source"] = src

        span, W = s["soak_span_s"], s["soak_iterations"]
        if not (span and W):
            continue
        s["max_possible_gain"] = W / (span / ref) - 1.0
        continuous_rate = 1.0 / ref
        best = None
        for row, c in zip(s["_duty"], s["_curve"]):
            T = row["cooldown_s"]
            r = c["recovery"]
            if r is None:
                continue
            # Recovery is measured against the session's own soak, which for a
            # short burst spans a smaller throughput range; rescale it onto the
            # continuous-training reference so r=1 still means "at cold rate".
            steady_work = span / ref
            rr = max(0.0, min(1.0, r))
            work = steady_work + rr * (W - steady_work)
            row["throughput_vs_continuous"] = (work / (span + T)) / continuous_rate
            if best is None or row["throughput_vs_continuous"] > best["throughput_vs_continuous"]:
                best = row
        if best:
            s["best_cooldown_s"] = best["cooldown_s"]
            s["best_throughput_vs_continuous"] = best["throughput_vs_continuous"]
            s["duty_cycle_verdict"] = (
                "duty_cycling_wins"
                if best["throughput_vs_continuous"] > 1.0 + DUTY_TOLERANCE
                else (
                    "continuous_wins"
                    if best["throughput_vs_continuous"] < 1.0 - DUTY_TOLERANCE
                    else "wash"
                )
            )


def print_selflimit_summary(s, args):
    print(f"\n=== SELF-LIMITING  {s['n_phases']} phase(s)"
          f"  session={s['bench_session_id'][:8]}"
          f"  {'COMPLETE' if s['complete'] else 'IN PROGRESS'}")
    print(f"    cold ref      {fmt(s['cold_ref_seconds_per_iter'])} s/iter")
    ref = s["reference_plateau_seconds_per_iter"]
    print(f"    continuous    {fmt(ref)} s/iter   ({s['reference_plateau_source']})")
    print("    delay   iters   span   compute  effective   duty   vs cont  converged")
    for p in s["_phases"]:
        vs = (ref / p["effective_seconds_per_iter"]) if ref else None
        flag = "yes" if p["converged"] else f"NO (ramp, drift {p['drift_s_per_iter']:+.2f})"
        print(f"    {p['delay_s']:5.2f}s {p['iterations']:6d} {p['phase_span_s']/60:5.0f}m"
              f"   {fmt(p['compute_seconds_per_iter'])}   {fmt(p['effective_seconds_per_iter'])}"
              f"  {fmt(p['duty'], '.3f')}   {fmt(vs)}   {flag}")
    if s["exchange_rate_dc_dD"] is not None:
        er = s["exchange_rate_dc_dD"]
        print(f"    exchange rate dc/dD = {er:+.3f} s per s of delay"
              f"   (needs < -1.000 for pacing to pay)")
    if s["duty_cycle_verdict"]:
        verdict = {
            "duty_cycling_wins": "PACING WINS",
            "continuous_wins": "CONTINUOUS WINS",
            "wash": f"WASH (within +/-{DUTY_TOLERANCE:.0%})",
        }[s["duty_cycle_verdict"]]
        print(f"    verdict       best={fmt(s['best_vs_continuous'])}x at "
              f"D={fmt(s['best_delay_s'], '.2f')}s  -> {verdict}")
    if not any(p["converged"] for p in s["_phases"]):
        print("    WARNING: no phase reached equilibrium — every value is a ramp,"
              " and ramps FLATTER pacing (compute still climbing)")


def print_cycle_summary(s, args):
    burst_min = (s["burst_seconds"] or 0) / 60.0
    print(f"\n=== SUSTAINED CYCLING  {burst_min:.0f}min on / {s['rest_seconds']:.0f}s off"
          f"  x{s['cycles_observed']}  session={s['bench_session_id'][:8]}"
          f"  {'COMPLETE' if s['complete'] else 'IN PROGRESS'}")
    print(f"    cold ref      {fmt(s['cold_ref_seconds_per_iter'])} s/iter")
    print("    per burst     "
          + "  ".join(str(c["iterations"]) for c in s["_cycles"]) + "  iters")
    print("                  "
          + "  ".join(f"{c['seconds_per_iter']:.2f}" for c in s["_cycles"]) + "  s/iter")
    print(f"    first burst   {s['first_burst_iterations']} iters"
          f"   settled {fmt(s['settled_burst_iterations_mean'], '.1f')} iters"
          f"  @ {fmt(s['settled_seconds_per_iter'])} s/iter")
    print(f"    setup cost    {fmt(s['setup_overhead_s_mean'], '.1f')} s/burst"
          f"   (restart overhead — measured, not assumed)")
    ref = s["reference_plateau_seconds_per_iter"]
    print(f"    continuous    {fmt(ref)} s/iter = {fmt(1/ref, '.5f') if ref else '—'} iter/s"
          f"   ({s['reference_plateau_source']})")
    print(f"    sustained     {fmt(s['sustained_iter_per_s'], '.5f')} iter/s"
          f"  = {fmt(s['sustained_vs_continuous'])}x continuous"
          f"   (cycles 2-{s['cycles_observed']}, rest gaps included)")
    print(f"    in-burst      {fmt(s['inburst_iter_per_s'], '.5f')} iter/s"
          f"  = {fmt(s['inburst_vs_continuous'])}x continuous"
          f"   (all restart cost forgiven)")
    if s["duty_cycle_verdict"]:
        verdict = {
            "duty_cycling_wins": "DUTY-CYCLING WINS",
            "continuous_wins": "CONTINUOUS WINS",
            "wash": f"WASH (within +/-{DUTY_TOLERANCE:.0%} noise)",
        }[s["duty_cycle_verdict"]]
        print(f"    verdict       {verdict}")
    if args.curve and s["_rest_probes"]:
        print("      recovery reached during each rest gap:")
        for r in s["_rest_probes"]:
            print(f"        after burst {r['after_cycle'] + 1}: "
                  f"{r['seconds_per_iter']:.3f} s/iter at +{r['rest_elapsed_s']:.0f}s")


def fmt(v, spec=".3f", dash="—"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", help="train_bench_metrics_thermal.jsonl pull(s)")
    ap.add_argument("-o", "--out", help="write aggregate JSON here")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--curve", action="store_true", help="print the full per-probe recovery curve"
    )
    args = ap.parse_args()

    commit, dirty = git_provenance()
    print(
        f"[thermal_aggregate] h10 | files={len(args.jsonl)} | commit={commit}"
        f"{' (dirty)' if dirty else ''} | host={socket.gethostname()} | "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

    if args.out and os.path.exists(args.out) and not args.overwrite:
        print(f"refusing to overwrite {args.out} (pass --overwrite)", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(args.jsonl)
    if not rows:
        print("no records", file=sys.stderr)
        sys.exit(1)

    sessions = {}
    for r in rows:
        sessions.setdefault(r.get("bench_session_id", "unknown"), []).append(r)

    summaries = [summarize_session(sid, rs) for sid, rs in sessions.items()]
    summaries.sort(key=lambda s: s.get("timestamp_utc") or "")
    apply_reference_plateau(summaries)

    for s in summaries:
        if s.get("session_type") == "cycle":
            print_cycle_summary(s, args)
            continue
        if s.get("session_type") == "selflimit":
            print_selflimit_summary(s, args)
            continue
        soak_min = (s["soak_seconds"] or 0) / 60.0
        label = f"soak={soak_min:.0f}min probe={s['probe_interval_s']}s"
        print(f"\n=== {label}  session={s['bench_session_id'][:8]}"
              f"  {'COMPLETE' if s['complete'] else 'IN PROGRESS'}")
        print(f"    cold ref      {fmt(s['cold_ref_seconds_per_iter'])} s/iter"
              f"  (n={s['cold_ref_n_iters']})")
        steady = ("" if s["plateau_is_steady_state"]
                  else f"  [too short for steady state; continuous ref = "
                       f"{fmt(s['reference_plateau_seconds_per_iter'])} from "
                       f"{s['reference_plateau_source']}]")
        print(f"    soak plateau  {fmt(s['soak_plateau_seconds_per_iter'])} s/iter"
              f"  (n={s['soak_plateau_n_iters']} of {s['soak_n_iters']}){steady}")
        print(f"    R (cold/hot)  {fmt(s['cold_hot_ratio_R'], '.3f')}x"
              f"   probes={s['n_probes']}")
        ms = "  ".join(
            f"t{int(m * 100)}={fmt(s[f't{int(m * 100)}_s'], '.0f')}s" for m in MILESTONES
        )
        print(f"    recovery      {ms}")
        print(f"    enum nominal  {fmt(s['enum_first_nominal_s'], '.0f')}s after soak"
              f"   (t90 lag {fmt(s['enum_vs_t90_gap_s'], '+.0f')}s)")
        if s["best_throughput_vs_continuous"] is not None:
            verdict = {
                "duty_cycling_wins": "DUTY-CYCLING WINS",
                "continuous_wins": "continuous wins",
                "wash": f"WASH (within +/-{DUTY_TOLERANCE:.0%} noise)",
            }[s["duty_cycle_verdict"]]
            print(f"    burst work    {s['soak_iterations']} iters in "
                  f"{fmt(s['soak_span_s'], '.0f')}s from cold  "
                  f"(cold-start bonus {fmt(s['max_possible_gain'], '+.1%')} "
                  f"= ceiling on any schedule)")
            print(f"    duty cycle    best={fmt(s['best_throughput_vs_continuous'])}x "
                  f"continuous at T={fmt(s['best_cooldown_s'], '.0f')}s  → {verdict}")
            print(f"      (probe-bound estimator, NOT the verdict — ignores "
                  f"re-throttling: {fmt(s['best_throughput_vs_continuous_probe_bound'])}x "
                  f"at T={fmt(s['best_cooldown_s_probe_bound'], '.0f')}s)")
        if s["stayed_charging"] is False:
            print("    WARNING: device was not charging for the whole run")

        if args.curve:
            print("      t(s)   s/iter  recovery  thermal")
            for c in s["_curve"]:
                print(f"      {c['cooldown_elapsed_s']:6.0f} {c['seconds_per_iter']:8.3f}"
                      f" {fmt(c['recovery'], '8.3f')}  {c['thermal_state']}")

    # A single-burst session can only ever bound a burst schedule from above;
    # if a sustained-cycling arm was actually run, IT is the answer. Say so,
    # loudly, so nobody quotes the optimistic figure.
    measured = [s for s in summaries if s.get("session_type") == "cycle"
                and s.get("sustained_vs_continuous")]
    claimed = [s for s in summaries if s.get("session_type") != "cycle"
               and s.get("duty_cycle_verdict") == "duty_cycling_wins"]
    if measured and claimed:
        m = measured[0]
        print(f"\n*** NOTE: {len(claimed)} single-burst session(s) above report "
              f"'DUTY-CYCLING WINS'. That is an UPPER BOUND extrapolated from one "
              f"burst.\n    The sustained arm actually ran the schedule and measured "
              f"{m['sustained_vs_continuous']:.3f}x — continuous training wins. Quote "
              f"the measured figure.")

    if args.out:
        flat = [{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries]
        payload = {
            "sessions": flat,
            "curves": {
                s["bench_session_id"]: s["_curve"]
                for s in summaries
                if s.get("session_type") not in ("cycle", "selflimit")
            },
            "duty_cycle": {
                s["bench_session_id"]: s["_duty"]
                for s in summaries
                if s.get("session_type") not in ("cycle", "selflimit")
            },
            "selflimit_phases": {
                s["bench_session_id"]: s["_phases"]
                for s in summaries
                if s.get("session_type") == "selflimit"
            },
            "cycles": {
                s["bench_session_id"]: s["_cycles"]
                for s in summaries
                if s.get("session_type") == "cycle"
            },
            "cycle_rest_probes": {
                s["bench_session_id"]: s["_rest_probes"]
                for s in summaries
                if s.get("session_type") == "cycle"
            },
            "git_commit": commit,
            "git_dirty": dirty,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
