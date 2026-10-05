"""Shared style for the E2E on-device-training figures (paper, PDF).

Palette choices (per the dataviz method):
  - USER identity = categorical, Okabe-Ito (pre-validated colorblind-safe;
    assigned in a FIXED per-user order, never cycled).
  - THERMAL = status, the reserved nominal/fair/serious/critical ramp, matching
    the existing eval/plot_thermal_stress.py figure so the paper stays consistent.
Labels use a t-shirt-size scheme keyed to profile size (the actual independent
variable), ascending XS -> XXL. Real 6 users, sorted by LaMP-3 profile size.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

# fingerprint -> short label used across all figures, ascending by profile size
# (405, 448, 500, 550, 653, 987). u00013218 (XL, 653) has no completed run yet.
USER_LABELS = {
    "u00008075": "XS", "u00005228": "S", "u00011077": "M",
    "u00005020": "L", "u00013218": "XL", "u00012502": "XXL",
}

# condition code -> human label, per the E2E plan's 4-condition matrix
CONDITION_LABELS = {
    "C0": "", "C1": "low power", "C2": "unplugged", "C4": "game load",
}

# Okabe-Ito, assigned in fixed order (by the label order above). CVD-safe.
_OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
USER_COLORS = {fp: _OKABE_ITO[i] for i, fp in enumerate(USER_LABELS)}

# Reserved thermal status ramp (matches plot_thermal_stress.py).
THERMAL_COLORS = {
    "nominal": "#2ca02c", "fair": "#ff7f0e",
    "serious": "#d62728", "critical": "#7f0000", "unknown": "#7f7f7f",
}

TOKEN_SLOPE_S = 0.0183  # wall_s per token, fallback; prefer agg["derived"]["cost_fit_tokens"]


def token_slope(agg):
    """Token-fit slope from the aggregate if present, else the constant."""
    tf = (agg.get("derived") or {}).get("cost_fit_tokens")
    return tf["slope_s_per_token"] if tf else TOKEN_SLOPE_S


def apply_rc():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150,
        # Figures are downscaled when placed in the two-column layout, which
        # thins the strokes. Near-black at a heavier weight survives that and
        # survives a greyscale print.
        "font.weight": "normal",
        "axes.labelweight": "normal", "axes.titleweight": "normal",
        "figure.titleweight": "normal", "legend.title_fontsize": 8,
        # DejaVu Sans ships only 400/700, so "semibold" renders as full bold.
        # Weight stays regular and the near-black colour below is what keeps
        # the text from washing out when the figure is downscaled.
        "mathtext.default": "it",
        "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A",
        "xtick.color": "#1A1A1A", "ytick.color": "#1A1A1A",
        "axes.edgecolor": "#1A1A1A", "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    })


def load_agg(path):
    return json.loads(Path(path).read_text())


def real_completed(agg):
    """Completed runs on the fused model, keyed for plotting."""
    return [r for r in agg["runs"]
            if r.get("completed") and r.get("model") == "SmolLM3-3B-a1lamp-4bit"]


def label(fp):
    return USER_LABELS.get(fp, str(fp)[-4:])


def panel_title(fp, condition):
    cond_label = CONDITION_LABELS.get(condition, condition)
    return f"{label(fp)} ({cond_label})" if cond_label else label(fp)


def thermal_trajectory_panels(agg, extra=None):
    """One panel per user's C0 run (sorted by profile size), plus optional extra
    (fingerprint, condition) runs inserted right after that same user's C0 panel
    (e.g. a C2/unplugged run placed next to its own C0 run for direct comparison).

    Returns (panels, xmax, ymax) where each panel is (title, run) and xmax/ymax
    are the shared max across every panel returned (for callers that want a
    common axis scale; independent-axis callers can ignore these).
    """
    completed = real_completed(agg)
    c0 = [r for r in completed if str(r["condition"]).startswith("C0")]
    extra = extra or []
    seen, panels = set(), []
    for r in sorted(c0, key=lambda r: r["profile_size"]):
        fp = r["user_fingerprint"]
        if fp in seen:
            continue
        seen.add(fp)
        panels.append((panel_title(fp, "C0"), r))
        for efp, econd in extra:
            if efp != fp:
                continue
            er = next((x for x in completed
                       if x["user_fingerprint"] == efp and x["condition"] == econd), None)
            if er is not None:
                panels.append((panel_title(efp, econd), er))
    # any extras whose fingerprint has no C0 panel above go at the end
    for efp, econd in extra:
        if efp in seen:
            continue
        er = next((x for x in completed
                   if x["user_fingerprint"] == efp and x["condition"] == econd), None)
        if er is not None:
            panels.append((panel_title(efp, econd), er))

    def curve(r):
        return [c for c in r["loss_curve"] if c.get("elapsed_s") is not None]
    xmax = max(c["elapsed_s"] / 60 for _, r in panels for c in curve(r)) * 1.02
    ymax = max(c["iter_per_sec"] for _, r in panels for c in curve(r)) * 1.1
    return panels, xmax, ymax


def save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = str(out).rsplit(".", 1)[0] + ".png"
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"wrote {out}  +  {png}")
