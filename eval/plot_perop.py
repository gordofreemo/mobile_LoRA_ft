#!/usr/bin/env python3
"""Per-op/per-phase decomposition (h11) figures.

Fig A (always): stacked phase-share bars vs token count, cool and hot passes
side by side — where the time inside one LoRA training iteration goes, and
whether the split moves with sequence length or with heat. Each bar is
annotated with its decomposition-overhead ratio (Σphases / fused), because
barriers destroy cross-phase overlap and the shares are only as trustworthy as
that number is close to 1 — the same caveat MELT flags for vm_profiler.

Fig B (only with --kernels): one pie per captured phase with per-kernel slices,
the MELT Fig-8 analog. Tier 2's per-kernel table is read by hand out of Xcode's
Metal debugger (GUI-only) and transcribed into a small JSON:
    [{"phase": "forward", "kernel": "steel_gemm_...", "category": "dense gemm",
      "time_ms": 1.23}, ...]

Locked design: experiments/2026-08-04-ondevice-perop-h11-plan.md (pinned via
/grill_me 2026-08-04). Reuses the shared on-device figure style.

    .venv-mlx/bin/python eval/plot_perop.py \
        --agg results/ondevice_perop_smollm3_4bit_2026-08-XX.json \
        --out results/ondevice/figures/perop_2026-08-XX.pdf \
        [--kernels results/ondevice/perop_kernels_2026-08-XX.json]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from _e2e_plot_style import apply_rc, load_agg, save, plt

PHASES = ["data_prep", "graph_build", "forward", "backward", "optimizer", "readback"]

# Okabe-Ito, assigned in fixed phase order (CVD-safe). The two dominant phases
# (forward, backward) take the two most separable hues.
PHASE_COLORS = {
    "data_prep": "#E69F00",
    "graph_build": "#56B4E9",
    "forward": "#0072B2",
    "backward": "#D55E00",
    "optimizer": "#009E73",
    "readback": "#CC79A7",
}

PASS_TITLES = {
    "cool": "pass 1 — cool start",
    # h10d: the plateau takes 40-50 min and pass 2 starts ~20-30 min in, so
    # this is realistic mid-session heat, NOT thermal equilibrium.
    "hot": "pass 2 — hot (mid-session, not equilibrium)",
}


def fig_shares(agg, out):
    cells = [c for c in agg["cells"] if c.get("phase_shares")]
    passes = []
    for c in cells:
        if c["pass"] not in passes:
            passes.append(c["pass"])

    fig, axes = plt.subplots(
        1, len(passes), figsize=(3.9 * len(passes), 4.3), sharey=True, squeeze=False
    )
    for ax, pass_name in zip(axes[0], passes):
        rows = sorted(
            (c for c in cells if c["pass"] == pass_name), key=lambda c: c["target_tokens"]
        )
        xs = list(range(len(rows)))
        bottom = [0.0] * len(rows)
        for phase in PHASES:
            vals = [100.0 * (r["phase_shares"].get(phase) or 0.0) for r in rows]
            ax.bar(
                xs, vals, bottom=bottom, width=0.72, color=PHASE_COLORS[phase],
                edgecolor="white", linewidth=0.5, label=phase,
            )
            bottom = [b + v for b, v in zip(bottom, vals)]

        for x, r in zip(xs, rows):
            ratio = r.get("decomposition_overhead_ratio")
            if ratio is not None:
                ax.text(
                    x, 101.5, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=7,
                    color="#3A3A3A",
                )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["target_tokens"]) for r in rows])
        ax.set_xlabel("example length (tokens)")
        ax.set_title(PASS_TITLES.get(pass_name, pass_name))
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("share of one training iteration (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, title="phase",
    )
    lora = agg.get("lora") or {}
    fig.suptitle(
        f"{agg.get('model')} — LoRA r={lora.get('rank')} {lora.get('keys')}, "
        f"{lora.get('num_layers')} layers, GC per block, batch 1\n"
        "annotation = Σphases / fused (barrier overhead; shares are exact only as this → 1)",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


# Metal pipeline names carry their launch configuration (dtype, group size,
# tile shape, batch index); strip it so legend entries read like op names, the
# way MELT Fig 8 lists TVM ops. Variants that differ only in config (e.g. the
# float and bf16 affine_qmm_t tiles) merge into one slice.
_KERNEL_CFG = re.compile(
    r"_(nax|gs_\d+|b_\d+|bm\d+|bn\d+|bk\d+|wm\d+|wn\d+|alN_\w+|batch_\d+"
    r"|MN_\w+|K_\w+|bfloat16_t|bfloat16|float32|float|int32)(?=_|$)"
)

# Fixed, semantic base colours per category (kernels take shades of their
# category's hue, so the base-vs-LoRA contrast survives the per-kernel split).
_CATEGORY_COLORS = {
    "quantized dequant-matmul (base)": "#0072B2",
    "dense gemm (LoRA)": "#D55E00",
    "elementwise + copies": "#E69F00",
    "activation (SiLU)": "#009E73",
    "attention + norm + RoPE": "#CC79A7",
    "loss": "#56B4E9",
    "sqrt/square (Adam v-hat)": "#8C6BB1",
    "bias correction (scalar pow)": "#7F7F7F",
}
_SPARE_COLORS = ["#117733", "#882255", "#44AA99", "#DDCC77", "#332288"]

# Slices below this share of the phase total fold into "other", MELT's own
# treatment of its <1% long tail (Fig 2b).
_OTHER_BELOW_PCT = 0.5


def _short_kernel(name):
    return _KERNEL_CFG.sub("", re.sub(r"\s*\(\d+\)$", "", name))


def _shade(color, frac):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def fig_kernels(kernels, out):
    """MELT Fig-8 analog: one pie per captured phase, slices = individual
    kernels (config suffix stripped), percentage labels around the pie and a
    per-pie legend of kernel names.

    Each row carries EITHER `time_ms` or `cost_pct` — Xcode's shader profiler
    reports per-pipeline cost as a percentage of the capture, which is what the
    2026-08-04 transcription holds. Either works: pie slices are normalised
    within a phase regardless, and Tier 1 already supplies the absolute phase
    totals. Mixing the two within ONE phase would silently compare unlike
    units, so that is rejected rather than summed.
    """
    by_phase = defaultdict(lambda: defaultdict(float))
    kernel_cat = {}
    units, tokens, bwd_layers = {}, {}, {}
    for k in kernels:
        phase = k["phase"]
        if k.get("capture_tokens"):
            tokens[phase] = k["capture_tokens"]
        if k.get("capture_backward_layers"):
            bwd_layers[phase] = k["capture_backward_layers"]
        if "time_ms" in k and "cost_pct" in k:
            raise SystemExit(f"row for {phase}/{k.get('kernel')} has both time_ms and cost_pct")
        unit = "time_ms" if "time_ms" in k else "cost_pct"
        if units.setdefault(phase, unit) != unit:
            raise SystemExit(f"phase {phase!r} mixes time_ms and cost_pct rows")
        name = _short_kernel(k["kernel"])
        by_phase[phase][name] += float(k[unit])
        kernel_cat.setdefault(name, k.get("category") or "other")

    order = [p for p in ["forward", "backward", "optimizer"] if p in by_phase]
    order += [p for p in sorted(by_phase) if p not in order]

    spare_iter = iter(_SPARE_COLORS * 4)
    cat_color = {
        c: _CATEGORY_COLORS.get(c) or next(spare_iter)
        for c in dict.fromkeys(kernel_cat[n] for p in by_phase.values() for n in p)
    }

    # One colour per kernel, fixed ACROSS phases. Shade rank is taken from the
    # kernel's total over all phases, not its rank within one pie, so a kernel
    # that appears in several phases (affine_qmm_t is in both the forward pass
    # and the backward pass as checkpoint recomputation) keeps the same colour
    # everywhere and the reader can track it between pies.
    kernel_total = defaultdict(float)
    for p in by_phase.values():
        for n, v in p.items():
            kernel_total[n] += v
    kernel_color, _cat_rank = {}, defaultdict(int)
    for n in sorted(kernel_total, key=lambda k: (kernel_cat[k], -kernel_total[k])):
        cat = kernel_cat[n]
        kernel_color[n] = _shade(cat_color[cat], min(0.55, 0.18 * _cat_rank[cat]))
        _cat_rank[cat] += 1

    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 3.6), squeeze=False)
    for ax, phase in zip(axes[0], order):
        total = sum(by_phase[phase].values()) or 1.0
        ranked = sorted(by_phase[phase].items(), key=lambda kv: -kv[1])
        named = [(n, v) for n, v in ranked if 100.0 * v / total >= _OTHER_BELOW_PCT]
        tail = [(n, v) for n, v in ranked if 100.0 * v / total < _OTHER_BELOW_PCT]

        colors = [kernel_color[n] for n, _ in named]

        names = [n for n, _ in named]
        vals = [v for _, v in named]
        if tail:
            names.append(f"other ({len(tail)} kernels)")
            vals.append(sum(v for _, v in tail))
            colors.append("#CCCCCC")

        pcts = [100.0 * v / total for v in vals]
        wedges, _ = ax.pie(
            vals,
            labels=[f"{p:.1f}%" if p >= 2.0 else "" for p in pcts],
            labeldistance=1.1, startangle=90, counterclock=False, colors=colors,
            wedgeprops=dict(edgecolor="white", linewidth=0.6),
            textprops=dict(fontsize=13, color="#1A1A1A"), normalize=True,
        )
        ax.legend(
            wedges, names, loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=12.5, handlelength=1.2, handleheight=1.2,
        )
        # Phases were captured at different token counts (the replay guest's
        # ~2 GB ceiling forced smaller captures for the heavier phases), so the
        # token count is part of each pie's identity, not a footnote; backward
        # is additionally a top-K-block subgraph sample, not the full phase.
        title = f"{phase} @{tokens[phase]} tok" if phase in tokens else phase
        if phase in bwd_layers:
            title += f" (top-{bwd_layers[phase]}-block sample)"
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
    fig.suptitle("Per-Kernel Share of Phase GPU Time (Metal Capture, One Iteration)",
                 fontsize=16, color="#1A1A1A")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.04, wspace=1.05)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True, help="output of eval/perop_aggregate.py")
    ap.add_argument("--out", default="results/ondevice/figures/perop.pdf")
    ap.add_argument(
        "--kernels",
        help="hand-transcribed Tier-2 per-kernel JSON; Fig B is skipped without it",
    )
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    fig_shares(agg, args.out)

    if args.kernels:
        kernels = json.loads(Path(args.kernels).read_text())
        base = str(args.out).rsplit(".", 1)[0]
        fig_kernels(kernels, f"{base}_kernels.pdf")
    else:
        print("no --kernels JSON given; skipping the Tier-2 kernel figure")


if __name__ == "__main__":
    main()
