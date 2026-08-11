#!/usr/bin/env python3
"""NAX off/on overlay of the h11 per-phase decomposition (the mechanism figure).

For each token count, two adjacent stacked bars (NAX off / NAX on) of the six
barriered phase times in absolute seconds — so both the level change (the
iteration shrinks) and the share change (backward's slice collapses) are
visible in one panel. Backward-share percentages annotated on the backward
segments. One pass per figure; default hot (the sustained-session regime h11
recommends quoting).

    .venv-mlx/bin/python eval/plot_perop_overlay.py \
        --off results/ondevice_perop_smollm3_4bit_2026-08-04.json \
        --on  results/ondevice_perop_smollm3_4bit_nax-on_<date>.json \
        --out results/ondevice/figures/perop_overlay_<date>.pdf
"""
import argparse

from _e2e_plot_style import apply_rc, load_agg, save, plt

PHASES = ["data_prep", "graph_build", "forward", "backward", "optimizer", "readback"]
# Okabe-Ito-adjacent ramp: backward gets the loudest color, forward second,
# everything else quiet grays (they are <1% combined).
PHASE_COLORS = {
    "data_prep": "#bbbbbb", "graph_build": "#999999", "forward": "#0072B2",
    "backward": "#D55E00", "optimizer": "#009E73", "readback": "#777777",
}


def cells_for_pass(agg, pass_name):
    out = {}
    for c in agg["cells"]:
        if c["pass"] == pass_name and c.get("phases"):
            out[c["target_tokens"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="NAX-off aggregate (2026-08-04 h11)")
    ap.add_argument("--on", required=True, help="NAX-on aggregate (rerun)")
    ap.add_argument("--pass", dest="pass_name", default="hot", choices=["cool", "hot"])
    ap.add_argument("--out", default="results/ondevice/figures/perop_overlay.pdf")
    args = ap.parse_args()
    apply_rc()

    off = cells_for_pass(load_agg(args.off), args.pass_name)
    on = cells_for_pass(load_agg(args.on), args.pass_name)
    tokens = sorted(set(off) & set(on))
    if not tokens:
        raise SystemExit("no common (pass, tokens) cells between the two aggregates")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.38
    for i, tok in enumerate(tokens):
        for j, (label, cells) in enumerate((("off", off), ("on", on))):
            c = cells[tok]
            x = i + (j - 0.5) * (width + 0.04)
            bottom = 0.0
            total = sum(c["phases"][p]["mean"] for p in PHASES)
            for p in PHASES:
                v = c["phases"][p]["mean"]
                ax.bar(x, v, width, bottom=bottom, color=PHASE_COLORS[p],
                       edgecolor="white", linewidth=0.3,
                       label=p if (i == 0 and j == 0) else None)
                if p == "backward" and v > 0.15:
                    ax.text(x, bottom + v / 2, f"{100 * v / total:.0f}%",
                            ha="center", va="center", fontsize=7, color="white")
                bottom += v
            ax.text(x, bottom + 0.06, label, ha="center", va="bottom", fontsize=7,
                    color="#3a3a3a")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([str(t) for t in tokens])
    ax.set_xlabel("target tokens")
    ax.set_ylabel("seconds per iteration (barriered, phase-stacked)")
    ax.set_title(f"One training iteration, NAX off vs on ({args.pass_name} pass)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save(fig, args.out)


if __name__ == "__main__":
    main()
