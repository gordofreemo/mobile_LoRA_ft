#!/usr/bin/env python3
"""
Aggregate Phase-3 E2E on-device per-user training telemetry
(Documents/train_bench_metrics_e2e.jsonl pulled off the phone) into a per-run
master table + derived cost/condition/variance/fidelity blocks.

Mirrors eval/bench_aggregate.py: stdlib only (no pandas on the Mac MLX venv),
flat JSON records in, descriptive stats out. Design:
experiments/2026-07-03-ondevice-e2e-training-plan.md (deliverables §).

Record types in the JSONL (h5 schema v2), grouped by bench_session_id = one run:
  run_start | train (per stepsPerReport window) | battery (30s timer) | run_end/error

Per-run summary → derived:
  P1 cost  : wall_time_s vs profile_size (least-squares fit) → extrapolate the
             sum over the real top-100 profile distribution ("cost to train 100").
  P3 cond  : C1/C2/C4 multipliers vs the SAME user's C0 (time, iter/s, peak, drain).
  variance : CV of wall-time & iter/s across a user's repeated C0 runs (M×3).
  P4 fidel : per-run on-device loss trajectory (x = epoch = step/n_user) + the
             cluster R5 curve if results/cluster_r5_losscurves/metrics_<fp>.jsonl
             exists. NOTE: on-device loss is FULL-SEQUENCE, cluster loss is
             ASSISTANT-ONLY — absolute values are NOT comparable; compare the
             convergence/shape. Emitted so the plot can pick axes accordingly.
  P5 energy: battery-level trajectory (meaningful only unplugged / charging=false).

Usage:
    python eval/e2e_aggregate.py results/ondevice/train_bench_metrics_e2e_*.jsonl \
        --out results/ondevice_e2e_smollm3_a1lamp_$(date +%Y-%m-%d).json
"""
import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

TOP100 = "data/lamp_user_stats/LaMP_3_top100_users.json"
CLUSTER_LOSS_DIR = "results/cluster_r5_losscurves"
# Users whose condition runs are read as C0-relative (label prefix match, so
# "C0", "C0-verify", "C0-smoke" all count as the C0 family — the aggregator
# separates smoke via the iterations_total / model fields, see is_real_run).
C0_PREFIX = "C0"

# h9 (2026-07-26): energy characterization. iPhone 17 Pro rated capacity,
# confirmed via test-device model number MG8N4ZD/A — the "ZD/A" order-number
# suffix is Germany/Austria/Switzerland/Benelux/France, which ships the
# physical-SIM-tray variant (NOT the US-only eSIM 4252 mAh variant). Nominal
# voltage is Apple's typical recent-iPhone Li-ion figure, not independently
# datasheet-confirmed for this exact model — see
# experiments/2026-07-26-ondevice-energy-h9-plan.md.
BATTERY_CAPACITY_MAH = 3998
BATTERY_NOMINAL_VOLTAGE_V = 3.87
FULL_BATTERY_JOULES = BATTERY_CAPACITY_MAH / 1000 * BATTERY_NOMINAL_VOLTAGE_V * 3600


def compute_energy(batt_curve):
    """Gross energy (J) + avg power (W) drained over this run's own
    UNPLUGGED battery samples only (the first sample of every h5/h9 run
    reads charging=True for an instant right after the USB launch — real
    duration/drain is measured between the first and last genuinely
    unplugged samples). None if fewer than 2 such samples exist."""
    unplugged = [b for b in batt_curve if b["charging"] is False
                 and isinstance(b["level"], (int, float)) and b["level"] >= 0
                 and b.get("elapsed_s") is not None]
    if len(unplugged) < 2:
        return None
    start_level, end_level = unplugged[0]["level"], unplugged[-1]["level"]
    duration_s = unplugged[-1]["elapsed_s"] - unplugged[0]["elapsed_s"]
    if duration_s <= 0:
        return None
    drained_frac = start_level - end_level
    gross_j = drained_frac * FULL_BATTERY_JOULES
    return {
        "start_level": start_level, "end_level": end_level,
        "duration_s": duration_s, "drained_frac": drained_frac,
        "gross_j": gross_j, "avg_power_w": gross_j / duration_s,
    }


def load_records(paths):
    recs = []
    # The device JSONL is cumulative, so successive (esp. dated) pulls overlap.
    # Dedup by exact line content — a record pulled twice is byte-identical
    # (timestamp_utc + session_id + step make distinct records never collide),
    # so this is safe and prevents double-counting across overlapping files.
    seen = set()
    for p in paths:
        for ln, line in enumerate(Path(p).read_text().splitlines(), 1):
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skip {p}:{ln}: {e}", file=sys.stderr)
    return recs


def describe(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals), "n": len(vals),
    }


def linfit(xs, ys):
    """Ordinary least squares y = a*x + b. Returns (slope, intercept, r)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    a = sxy / sxx
    b = my - a * mx
    syy = sum((y - my) ** 2 for y in ys)
    r = (sxy / (sxx * syy) ** 0.5) if syy > 0 else None
    return a, b, r


def summarize_run(session_recs):
    """One run (bench_session_id) → summary dict. Handles both a real
    training run (run_start/train/battery/run_end|error) and an h9 idle
    energy baseline (idle_baseline_start/idle_baseline/idle_baseline_end,
    no training at all) — the latter has no n_user/profile_size/model/etc.,
    which just come through as None below."""
    start = next((r for r in session_recs if r["record_type"] == "run_start"), None)
    end = next((r for r in session_recs
                if r["record_type"] in ("run_end", "error", "idle_baseline_end")), None)
    train = [r for r in session_recs if r["record_type"] == "train"]
    # h9: idle_baseline periodic samples carry the same battery/thermal/cpu
    # fields as a training run's "battery" samples, just under a different
    # record_type — treat them identically here.
    batt = [r for r in session_recs if r["record_type"] in ("battery", "idle_baseline")]
    is_baseline = any(r["record_type"] == "idle_baseline_start" for r in session_recs)
    meta = start or (train[0] if train else session_recs[0])

    n_user = meta.get("n_user")
    prof = meta.get("profile_size")
    # loss trajectory with epoch = step / n_user (comparable to cluster's epoch)
    loss_curve = [
        {"step": t["step"], "epoch": (t["step"] / n_user) if n_user else None,
         "loss": t["training_loss"], "iter_per_sec": t["iter_per_sec"],
         "tok_per_sec": t.get("tok_per_sec"), "elapsed_s": t.get("elapsed_s"),
         "thermal_state": t.get("thermal_state"),
         "peak_mem_bytes": t.get("peak_mem_bytes")}
        for t in train
    ]
    # battery trajectory (level over elapsed); drain only if a discharging sample exists
    batt_curve = [{"elapsed_s": b.get("elapsed_s"), "level": b.get("battery_level"),
                   "charging": b.get("charging"), "cpu_util_pct": b.get("cpu_util_pct")}
                  for b in batt]
    unplugged = [b for b in batt_curve if b["charging"] is False and isinstance(b["level"], (int, float)) and b["level"] >= 0]
    drain_pct = None
    if len(unplugged) >= 2:
        drain_pct = (unplugged[0]["level"] - unplugged[-1]["level"]) * 100.0

    wall = None
    if end and isinstance(end.get("elapsed_s"), (int, float)):
        wall = end["elapsed_s"]
    elif train:
        wall = train[-1].get("elapsed_s")
    elif batt_curve and batt_curve[-1].get("elapsed_s") is not None:
        # h9: a run that died silently (no run_end/error, e.g. the L profile
        # dying of battery exhaustion) still has its last surviving battery
        # sample's elapsed_s as the best available "how long did it run" figure.
        wall = batt_curve[-1]["elapsed_s"]

    peak = max([t["peak_mem_bytes"] for t in train] + [b.get("peak_mem_bytes", 0) for b in batt] or [0])
    completed = bool(end and end["record_type"] == "run_end" and end.get("adapter_saved"))

    # tokens/step ≈ tok_per_sec / iter_per_sec (batch 1) — the post-cap mean
    # sequence length, which (× iterations) is the REAL cost driver: it varies
    # per user independent of profile_size (a user with long reviews costs far
    # more per step). See the S(531)-vs-448(947) tokens/step gap.
    tok_per_step = [t["tok_per_sec"] / t["iter_per_sec"]
                    for t in train if t.get("iter_per_sec")]
    mean_tok_per_step = statistics.fmean(tok_per_step) if tok_per_step else None
    total_tokens_est = (mean_tok_per_step * meta.get("iterations_total")
                        if mean_tok_per_step and meta.get("iterations_total") else None)

    return {
        "bench_session_id": meta.get("bench_session_id"),
        "bench_schema_version": meta.get("bench_schema_version"),
        "user_fingerprint": meta.get("user_fingerprint"),
        "condition": meta.get("condition"),
        "profile_size": prof,
        "n_user": n_user,
        "iterations_total": meta.get("iterations_total"),
        "epochs": meta.get("epochs"),
        "model": meta.get("model"),
        "app_build": meta.get("app_build"),
        "git_commit": meta.get("git_commit"),
        "completed": completed,
        "adapter_saved": bool(end.get("adapter_saved")) if end else False,
        "error": (end or {}).get("error") if end and end["record_type"] == "error" else None,
        "wall_time_s": wall,
        "n_windows": len(train),
        "mean_iter_per_sec": describe([t["iter_per_sec"] for t in train])["mean"],
        "mean_tok_per_sec": describe([t["tok_per_sec"] for t in train])["mean"],
        "mean_tok_per_step": mean_tok_per_step,
        "total_tokens_est": total_tokens_est,
        "peak_mem_bytes": peak or None,
        "first_loss": train[0]["training_loss"] if train else None,
        "final_loss": train[-1]["training_loss"] if train else None,
        "end_thermal": train[-1]["thermal_state"] if train else None,
        "low_power_mode": train[-1]["low_power_mode"] if train else meta.get("low_power_mode"),
        "battery_drain_pct_unplugged": drain_pct,
        "loss_curve": loss_curve,
        "battery_curve": batt_curve,
        # h9 additions:
        "is_baseline": is_baseline,
        "energy": compute_energy(batt_curve),
        "mean_cpu_util_pct": describe(
            [b.get("cpu_util_pct") for b in batt if isinstance(b.get("cpu_util_pct"), (int, float))]
        )["mean"],
    }


def is_real_run(run):
    """A full to-completion run on the FUSED model (excludes base-model smokes
    and short --max-iters checks). Real = fused model AND iterations == 3*n_user."""
    if run.get("model") != "SmolLM3-3B-a1lamp-4bit":
        return False
    it, nu, ep = run.get("iterations_total"), run.get("n_user"), run.get("epochs") or 3
    return bool(nu) and it == ep * nu


def load_cluster_curves(users):
    out = {}
    for u in users:
        p = Path(CLUSTER_LOSS_DIR) / f"metrics_{u}.jsonl"
        if not p.exists():
            continue
        pts = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "loss" in d and "epoch" in d:
                pts.append({"epoch": d["epoch"], "loss": d["loss"], "step": d.get("step")})
        if pts:
            out[u] = pts
    return out


def top100_profiles():
    p = Path(TOP100)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return [u["profile_size"] for u in d.get("users", []) if "profile_size" in u]


def derive(runs):
    real = [r for r in runs if is_real_run(r) and r["completed"]]
    d: dict = {"n_runs_total": len(runs), "n_real_completed": len(real)}

    # ---- P1 cost: wall-time vs profile (one point per user; if a user has >1
    #      C0 run take the mean) over C0-family real runs ----
    c0 = [r for r in real if str(r["condition"]).startswith(C0_PREFIX)]
    by_user = {}
    for r in c0:
        by_user.setdefault(r["user_fingerprint"], []).append(r)
    cost_pts = []
    for u, rs in by_user.items():
        prof = rs[0]["profile_size"]
        wt = describe([r["wall_time_s"] for r in rs])["mean"]
        cost_pts.append({"user": u, "profile_size": prof, "wall_time_s": wt,
                         "mean_tok_per_step": describe([r["mean_tok_per_step"] for r in rs])["mean"],
                         "total_tokens_est": describe([r["total_tokens_est"] for r in rs])["mean"],
                         "n_c0_runs": len(rs)})
    cost_pts.sort(key=lambda x: x["profile_size"])
    d["cost_points_c0"] = cost_pts

    # profile-size fit (noisy — ignores per-user sequence length)
    fit = linfit([p["profile_size"] for p in cost_pts], [p["wall_time_s"] for p in cost_pts]) if len(cost_pts) >= 2 else None
    if fit:
        a, b, r = fit
        profs = top100_profiles()
        total_s = sum(a * pf + b for pf in profs) if profs else None
        d["cost_fit"] = {
            "predictor": "profile_size",
            "slope_s_per_profile_entry": a, "intercept_s": b, "pearson_r": r,
            "extrapolated_total_100_device_hours": (total_s / 3600.0) if total_s else None,
            "n_top100_users": len(profs) if profs else None,
        }
    else:
        d["cost_fit"] = None

    # total-tokens fit (the real cost law: wall ≈ s/token · total_tokens). Better
    # predictor than profile_size. Extrapolation to 100 needs each user's token
    # count (profile_size × their post-cap mean seq len) — computable only after
    # building all 100 users' data; flagged rather than guessed here.
    tok_pts = [p for p in cost_pts if p.get("total_tokens_est")]
    tfit = linfit([p["total_tokens_est"] for p in tok_pts], [p["wall_time_s"] for p in tok_pts]) if len(tok_pts) >= 2 else None
    if tfit:
        a, b, r = tfit
        d["cost_fit_tokens"] = {
            "predictor": "total_tokens_est",
            "slope_s_per_token": a, "intercept_s": b, "pearson_r": r,
            "note": "extrapolation to 100 needs per-user token counts (build all 100 users' data first)",
        }
    else:
        d["cost_fit_tokens"] = None

    # ---- P3 condition multipliers vs same-user C0 ----
    mult = []
    for u, rs in by_user.items():
        c0_wt = describe([r["wall_time_s"] for r in rs])["mean"]
        c0_ips = describe([r["mean_iter_per_sec"] for r in rs])["mean"]
        c0_peak = describe([r["peak_mem_bytes"] for r in rs])["mean"]
        for other in [x for x in real if x["user_fingerprint"] == u and not str(x["condition"]).startswith(C0_PREFIX)]:
            mult.append({
                "user": u, "condition": other["condition"],
                "time_x": (other["wall_time_s"] / c0_wt) if c0_wt else None,
                "iter_per_sec_x": (other["mean_iter_per_sec"] / c0_ips) if c0_ips else None,
                "peak_mem_x": (other["peak_mem_bytes"] / c0_peak) if c0_peak else None,
                "battery_drain_pct_unplugged": other["battery_drain_pct_unplugged"],
            })
    d["condition_multipliers"] = mult

    # ---- variance: repeated C0 runs per user ----
    var = {}
    for u, rs in by_user.items():
        if len(rs) >= 2:
            wt = describe([r["wall_time_s"] for r in rs])
            ips = describe([r["mean_iter_per_sec"] for r in rs])
            var[u] = {
                "n": len(rs),
                "wall_time_cv": (wt["std"] / wt["mean"]) if wt["mean"] else None,
                "iter_per_sec_cv": (ips["std"] / ips["mean"]) if ips["mean"] else None,
                "wall_time": wt, "iter_per_sec": ips,
            }
    d["variance_c0"] = var

    # ---- h9: energy (joules), baseline-corrected where a paired idle
    #      baseline exists for the same user; falls back to any other
    #      available baseline (flagged as a proxy) otherwise. Deliberately
    #      NOT gated on `completed` — a run that died mid-training (e.g. the
    #      L profile, battery-exhaustion death) still gets an energy figure
    #      for whatever it actually drained before dying. ----
    # h9-only (schema v3+) — excludes pre-h9 runs (schema v2 and earlier)
    # recorded before this round's harness/energy code existed, which would
    # otherwise get spuriously paired with an h9 baseline from a different
    # day/environmental setup (airplane mode, brightness, etc. weren't part
    # of the original E2E backlog's protocol).
    h9_runs = [r for r in runs if (r.get("bench_schema_version") or 0) >= 3]
    baselines = {r["user_fingerprint"]: r for r in h9_runs if r["is_baseline"] and r["energy"]}
    energy_pts = []
    for r in h9_runs:
        if r["is_baseline"] or not r["energy"] or not is_real_run(r):
            continue
        gross_j = r["energy"]["gross_j"]
        base = baselines.get(r["user_fingerprint"])
        baseline_source = "matched"
        if base is None and baselines:
            base = next(iter(baselines.values()))
            baseline_source = "proxy"
        elif base is None:
            baseline_source = "none"
        baseline_avg_w = base["energy"]["avg_power_w"] if base else 0.0
        net_j = gross_j - baseline_avg_w * r["energy"]["duration_s"]
        energy_pts.append({
            "user": r["user_fingerprint"], "profile_size": r["profile_size"],
            "condition": r["condition"], "completed": r["completed"],
            "iterations_total": r["iterations_total"],
            "iterations_reached": (r["loss_curve"][-1]["step"] if r["loss_curve"] else None),
            "duration_s": r["energy"]["duration_s"],
            "gross_j": gross_j, "net_j": net_j,
            "avg_power_w": r["energy"]["avg_power_w"],
            "pct_full_battery_gross": 100.0 * gross_j / FULL_BATTERY_JOULES,
            "pct_full_battery_net": 100.0 * net_j / FULL_BATTERY_JOULES,
            "baseline_source": baseline_source,
            "baseline_user": base["user_fingerprint"] if base else None,
        })
    energy_pts.sort(key=lambda x: x["profile_size"] or 0)
    d["energy_h9"] = {
        "full_battery_joules": FULL_BATTERY_JOULES,
        "battery_capacity_mah": BATTERY_CAPACITY_MAH,
        "battery_nominal_voltage_v": BATTERY_NOMINAL_VOLTAGE_V,
        "baselines": {u: r["energy"] for u, r in baselines.items()},
        "points": energy_pts,
    }
    return d


def fmt(x, nd=1):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_report(agg):
    print("=" * 92)
    print(f"E2E on-device training — {agg['n_records']} records, "
          f"{len(agg['runs'])} runs ({agg['derived']['n_real_completed']} real+completed)")
    print(f"model={agg['model']}  device={agg['device_model']}  os={agg['os_version']}")
    print("=" * 92)
    hdr = ("user", "cond", "prof", "iters", "wall_s", "iter/s", "tok/stp", "peak_MB", "loss0→lossN", "thrm", "done")
    print("{:<11}{:<7}{:>5}{:>7}{:>8}{:>8}{:>8}{:>9}  {:<15}{:<8}{:<4}".format(*hdr))
    for r in sorted(agg["runs"], key=lambda x: (str(x["user_fingerprint"]), str(x["condition"]))):
        loss = f"{fmt(r['first_loss'],2)}→{fmt(r['final_loss'],2)}"
        peak = (r["peak_mem_bytes"] // 1048576) if r["peak_mem_bytes"] else None
        print("{:<11}{:<7}{:>5}{:>7}{:>8}{:>8}{:>8}{:>9}  {:<15}{:<8}{:<4}".format(
            str(r["user_fingerprint"])[-8:], str(r["condition"])[:6],
            fmt(r["profile_size"]), fmt(r["iterations_total"]),
            fmt(r["wall_time_s"], 0), fmt(r["mean_iter_per_sec"], 3),
            fmt(r["mean_tok_per_step"], 0),
            fmt(peak), loss, str(r["end_thermal"])[:7], "Y" if r["completed"] else "-"))
    dv = agg["derived"]
    if dv.get("cost_fit"):
        cf = dv["cost_fit"]
        print("-" * 92)
        print(f"cost fit (profile): wall_s ≈ {fmt(cf['slope_s_per_profile_entry'],2)}·profile "
              f"+ {fmt(cf['intercept_s'],0)}  (r={fmt(cf['pearson_r'],3)})")
        if cf.get("extrapolated_total_100_device_hours"):
            print(f"  → extrapolated cost to train all {cf['n_top100_users']} users: "
                  f"{fmt(cf['extrapolated_total_100_device_hours'],1)} device-hours "
                  f"(profile-only — see token fit)")
    if dv.get("cost_fit_tokens"):
        tf = dv["cost_fit_tokens"]
        print(f"cost fit (tokens):  wall_s ≈ {fmt(tf['slope_s_per_token'],4)}·total_tokens "
              f"+ {fmt(tf['intercept_s'],0)}  (r={fmt(tf['pearson_r'],3)}) ← real cost law")
    for m in dv.get("condition_multipliers", []):
        print(f"cond {m['condition']} (user …{str(m['user'])[-6:]}): "
              f"time×{fmt(m['time_x'],2)} iter/s×{fmt(m['iter_per_sec_x'],2)} "
              f"peak×{fmt(m['peak_mem_x'],2)} drain={fmt(m['battery_drain_pct_unplugged'],1)}%")
    for u, v in dv.get("variance_c0", {}).items():
        print(f"variance …{u[-6:]} (n={v['n']}): wall CV={fmt(v['wall_time_cv'],3)} "
              f"iter/s CV={fmt(v['iter_per_sec_cv'],3)}")
    eh9 = dv.get("energy_h9")
    if eh9 and eh9.get("points"):
        print("-" * 92)
        print(f"h9 energy (full battery = {fmt(eh9['full_battery_joules'],0)} J, "
              f"{eh9['battery_capacity_mah']} mAh @ {eh9['battery_nominal_voltage_v']} V):")
        for p in eh9["points"]:
            done = "done" if p["completed"] else f"DIED@{fmt(p['iterations_reached'],0)}/{fmt(p['iterations_total'],0)}"
            print(f"  …{str(p['user'])[-6:]} prof={fmt(p['profile_size'],0)} [{done}]: "
                  f"gross={fmt(p['gross_j'],0)}J ({fmt(p['pct_full_battery_gross'],1)}%batt)  "
                  f"net={fmt(p['net_j'],0)}J ({fmt(p['pct_full_battery_net'],1)}%batt)  "
                  f"baseline={p['baseline_source']}")
    if not dv.get("n_real_completed"):
        print("-" * 92)
        print("No real completed runs yet — table shows smokes/partials. Run the matrix, then re-aggregate.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="e2e JSONL file(s); default results/ondevice/train_bench_metrics_e2e_*.jsonl")
    ap.add_argument("--out", help="write aggregated JSON here")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/ondevice/train_bench_metrics_e2e_*.jsonl"))
    if not paths:
        print("No e2e telemetry files matched.", file=sys.stderr)
        sys.exit(1)
    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    # group by session
    sessions = {}
    for r in records:
        sessions.setdefault(r.get("bench_session_id"), []).append(r)
    runs = [summarize_run(rs) for rs in sessions.values()]
    runs.sort(key=lambda r: (str(r["user_fingerprint"]), str(r["condition"])))

    derived = derive(runs)
    users_seen = sorted({r["user_fingerprint"] for r in runs if r["user_fingerprint"]})
    cluster = load_cluster_curves(users_seen)

    agg = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in paths],
        "n_records": len(records),
        "app_builds": sorted({r.get("app_build") for r in records if r.get("app_build")}),
        "device_model": next((r.get("device_model") for r in records if r.get("device_model")), None),
        "os_version": next((r.get("os_version") for r in records if r.get("os_version")), None),
        "model": next((r.get("model") for r in records if r.get("model")), None),
        "runs": runs,
        "derived": derived,
        "cluster_loss_curves": cluster,  # for the P4 fidelity overlay
    }

    print_report(agg)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
