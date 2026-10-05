#!/usr/bin/env python3
"""Plot the sustained-decode (5-minute stress) throttle curve for an on-device
inference benchmark run, with the decode line coloured by the reported
``ProcessInfo.thermalState``.

Canonical source = the aggregate JSON (``results/ondevice_base_*.json``), whose
``stress.decay`` array carries, per 256-token segment, ``cumulative_tokens``,
``gen_tps`` and ``thermal_state``. Peak memory for the stress run is read from
the raw telemetry referenced in the aggregate's ``source_files``.

Single figure:
  python eval/plot_thermal_stress.py \
      --agg results/ondevice_base_qwen3_8b_4bit_2026-06-22.json \
      --model-label "Qwen3-8B-4bit" \
      --out overleaf/sections/figures/thermal_stress_qwen3_8b.pdf

Overlay (line colour = model; thermal shown as a marker edge):
  python eval/plot_thermal_stress.py --out .../thermal_stress_overlay.pdf \
      --overlay "results/ondevice_base_smollm3_4bit_2026-06-21.json:SmolLM3-3B-4bit:#1f77b4" \
                "results/ondevice_base_qwen3_8b_4bit_2026-06-22.json:Qwen3-8B-4bit:#d62728"
"""
import argparse
import json

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SEG_TOKENS = 256  # stress harness emits one telemetry record per 256-token segment

# ProcessInfo.thermalState enum -> colour (apple ordering: nominal<fair<serious<critical)
THERMAL_COLORS = {
    "nominal": "#2ca02c",
    "fair": "#ff7f0e",
    "serious": "#d62728",
    "critical": "#7f0000",
    "unknown": "#7f7f7f",
}
THERMAL_ORDER = ["nominal", "fair", "serious", "critical", "unknown"]


def load_decay(agg_path):
    """Return (agg_dict, cum_tokens, tps, thermal_states, elapsed_s) from the
    aggregate JSON. ``elapsed_s`` is real wall-clock time per sample when the
    telemetry carried it (capped-stress h4), else None (continuous stress)."""
    agg = json.load(open(agg_path))
    decay = agg["stress"]["decay"]
    cum = [d["cumulative_tokens"] for d in decay]
    tps = [d["gen_tps"] for d in decay]
    therm = [d.get("thermal_state", "unknown") for d in decay]
    elapsed = [d.get("elapsed_s") for d in decay]
    elapsed = elapsed if all(e is not None for e in elapsed) else None
    return agg, cum, tps, therm, elapsed


def elapsed_axis(tps, elapsed_s):
    """Real per-sample elapsed wall time when present, else reconstruct from tok/s
    (continuous stress: fixed SEG_TOKENS per segment, ignores prefill overhead)."""
    if elapsed_s is not None:
        return elapsed_s
    out, t = [], 0.0
    for v in tps:
        t += SEG_TOKENS / v
        out.append(t)
    return out


def stress_peak_gb(agg):
    """Peak memory of the stress run. Prefer the aggregate's stress block (h4),
    then the raw continuous-stress telemetry, then a grid-cell fallback."""
    stress = agg.get("stress") or {}
    pk = (stress.get("peak_mem_bytes") or {}).get("max")
    if pk:
        return pk / 1e9
    for src in agg.get("source_files", []):
        try:
            recs = [json.loads(l) for l in open(src)]
        except OSError:
            continue
        s = [r["peak_mem_bytes"] for r in recs if r.get("max_tokens") == 60000]
        if s:
            return max(s) / 1e9
    # fallback: max peak over grid cells
    return max(c["peak_mem_bytes"]["mean"] for c in agg["grid_cells"]) / 1e9


def floor_throttle(tps, plateau_frac):
    start = tps[0]
    n_plateau = max(1, int(len(tps) * plateau_frac))
    floor = sum(tps[-n_plateau:]) / n_plateau
    return start, floor, 100.0 * (1.0 - floor / start)


def thermal_legend(states_present):
    """Legend handles for the thermal states that actually occur, in enum order."""
    return [
        Line2D([0], [0], color=THERMAL_COLORS[s], lw=3,
               label=f"thermalState = {s}")
        for s in THERMAL_ORDER if s in states_present
    ]


def plot_single(agg_path, label, out, thermal_note, plateau_frac, title="Sustained Decode"):
    agg, cum_tok, tps, therm, elapsed_s = load_decay(agg_path)
    elapsed = elapsed_axis(tps, elapsed_s)
    start, floor, throttle_pct = floor_throttle(tps, plateau_frac)
    peak_gb = stress_peak_gb(agg)

    knee_i = next((i for i, v in enumerate(tps) if v <= floor * 1.05), len(tps) - 1)
    knee_t, knee_v = elapsed[knee_i], tps[knee_i]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # line coloured per segment by the thermal state recorded during that segment
    pts = list(zip(elapsed, tps))
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    seg_colors = [THERMAL_COLORS.get(therm[i + 1], THERMAL_COLORS["unknown"])
                  for i in range(len(pts) - 1)]
    ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=2.2, zorder=2))
    ax.scatter(elapsed, tps, c=[THERMAL_COLORS.get(t, THERMAL_COLORS["unknown"]) for t in therm],
               s=28, zorder=3, edgecolors="white", linewidths=0.5)

    ytop = start * 1.12
    knee_label_y = min(knee_v + 0.30 * start, ytop * 0.84)
    ax.annotate(f"knee ~{knee_t:.0f}s\n({knee_v:.1f} tok/s)",
                xy=(knee_t, knee_v), xytext=(knee_t + 0.06 * elapsed[-1], knee_label_y),
                arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9)
    ax.text(0.97, 0.93,
            f"{throttle_pct:.0f}% sustained throttle\n({start:.1f}$\\rightarrow${floor:.1f} tok/s)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    if thermal_note:
        ax.text(0.5, 0.30, thermal_note, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.4")

    ax.set_xlabel("Approximate elapsed time (s)")
    ax.set_ylabel("Decode speed (tok/s)")
    ax.set_xlim(0, max(elapsed) * 1.02)
    ax.set_ylim(0, ytop)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=thermal_legend(set(therm)), loc="lower left", fontsize=8)

    secax = ax.secondary_xaxis("top")
    step = max(1, len(elapsed) // 6)
    secax.set_xticks(elapsed[::step])
    secax.set_xticklabels([str(c) for c in cum_tok[::step]])
    secax.set_xlabel("Cumulative tokens generated")

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"  start={start:.1f} floor={floor:.1f} throttle={throttle_pct:.0f}% "
          f"knee~{knee_t:.0f}s total_tok={cum_tok[-1]} peak={peak_gb:.2f}GB "
          f"thermal={sorted(set(therm))}")


def overlay(specs, out, plateau_frac, title="Sustained Decode"):
    """specs: list of (agg_path, label, color). Both curves on shared axes;
    line colour distinguishes the model, marker fill encodes thermal state."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    states_present = set()
    xmax = 0.0
    for agg_path, label, color in specs:
        agg, cum_tok, tps, therm, elapsed_s = load_decay(agg_path)
        elapsed = elapsed_axis(tps, elapsed_s)
        start, floor, _ = floor_throttle(tps, plateau_frac)
        peak_gb = stress_peak_gb(agg)
        states_present |= set(therm)
        mins = elapsed[-1] / 60.0
        ax.plot(elapsed, tps, "-", color=color, lw=2, zorder=2,
                label=f"{label}  ({start:.1f}$\\rightarrow${floor:.1f} tok/s, "
                      f"peak {peak_gb:.1f} GB)")
        ax.scatter(elapsed, tps, s=34, zorder=3, linewidths=1.1, edgecolors=color,
                   c=[THERMAL_COLORS.get(t, THERMAL_COLORS["unknown"]) for t in therm])
        ax.annotate(f"{label}: {cum_tok[-1]:,} tok in {mins:.0f} min",
                    xy=(elapsed[-1], tps[-1]), xytext=(elapsed[-1] * 0.50, tps[-1] + 2.6),
                    color=color, fontsize=8)
        xmax = max(xmax, elapsed[-1])

    ax.set_xlabel("Approximate elapsed time (s)")
    ax.set_ylabel("Decode speed (tok/s)")
    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    model_handles = ax.get_legend_handles_labels()[0]
    handles = (model_handles
               + [Line2D([0], [0], color="none", label="")]
               + thermal_legend(states_present))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", help="aggregate JSON with a stress.decay block")
    ap.add_argument("--model-label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thermal-note", default="",
                    help="optional caption annotating the thermalState behaviour")
    ap.add_argument("--plateau-frac", type=float, default=0.5,
                    help="fraction of the run (from the end) treated as the throttled plateau")
    ap.add_argument("--title", default="Sustained Decode", help="plot title")
    ap.add_argument("--overlay", nargs="+", metavar="AGG:LABEL:COLOR",
                    help="produce a single overlay of multiple runs")
    args = ap.parse_args()

    if args.overlay:
        specs = [tuple(spec.split(":", 2)) for spec in args.overlay]
        overlay(specs, args.out, args.plateau_frac, args.title)
    else:
        plot_single(args.agg, args.model_label, args.out, args.thermal_note,
                    args.plateau_frac, args.title)


if __name__ == "__main__":
    main()
