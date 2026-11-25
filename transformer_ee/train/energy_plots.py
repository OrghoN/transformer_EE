# ------------------------- energy_plots.py ------------------------
"""
Energy-dependent accuracy of the model compared with the standard NOvA reconstructed-energy
estimators. MVtrainer draws these once per epoch on the validation set and overwrites them each
time, the same way it handles loss.png and accuracy_<target>.png.

For one target (e.g. trueLepE) the validation events are binned in TRUE energy. In each bin and
for every estimator (the transformer and each standard estimator) we plot

    median percent error = 100 * median of (E_est - E_true) / E_true

as a 1D histogram (one step line per estimator). 0 % means that half the events in the bin are
over-estimated and half are under-estimated.

Two binnings are drawn per target:
    full : the whole true-energy range of the validation set, 0 to max (coarse bins)
    lowE : 0 to 1.5 GeV with finer bins, where the magnetic-moment signal lives

Only events with E_true > 0 and a finite value for every estimator are used, so all estimators
are compared on exactly the same events. The statistics are unweighted (natural spectrum).
"""

import os
from typing import Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Standard estimators to compare against, per target. Only columns that are present in the
# config "eval_columns" are used. Override with the config key "reco_comparison".
DEFAULT_RECO_ESTIMATORS: Dict[str, List[str]] = {
    "trueLepE": ["nueRecoLepE", "shwCalE"],  # kEME_2024, and the analysis' leading-shower energy
    "trueE": ["nueRecoE"],                    # kNueEnergy2024
}

# Fixed colour (and line style) per estimator slot, in order: transformer first, then the
# standard estimators. The line style is a second encoding so the curves stay distinguishable
# without colour.
_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_LINESTYLES = ["-", "--", "-.", ":"]

MIN_EVENTS_PER_BIN = 20

N_BINS_FULL = 25        # bins over the full range [0, max true energy]
LOW_E_MAX_GEV = 1.5     # upper edge of the low-energy plot
LOW_E_BIN_GEV = 0.05    # bin width of the low-energy plot (30 bins)


def full_range_bin_edges(true: np.ndarray, n_bins: int = N_BINS_FULL) -> np.ndarray:
    """n_bins linear bins from 0 to the largest positive true value (rounded up to 0.5 GeV)."""
    t = true[np.isfinite(true) & (true > 0)]
    hi = float(t.max()) if t.size else 1.0
    hi = max(0.5, np.ceil(hi / 0.5) * 0.5)
    return np.linspace(0.0, hi, n_bins + 1)


def low_energy_bin_edges(e_max: float = LOW_E_MAX_GEV, width: float = LOW_E_BIN_GEV) -> np.ndarray:
    """Linear bins of the given width from 0 to e_max GeV."""
    n_bins = int(round(e_max / width))
    return np.linspace(0.0, e_max, n_bins + 1)


def binned_median_pct_error(true: np.ndarray, est: np.ndarray, edges: Sequence[float],
                            min_count: int = MIN_EVENTS_PER_BIN) -> Dict[str, np.ndarray]:
    """Per true-energy bin: number of events and median of 100 * (est - true) / true.
    Bins with fewer than min_count events get NaN. Events outside the edges are ignored."""
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    med = np.full(n_bins, np.nan)
    idx = np.digitize(true, edges) - 1
    idx[true == edges[-1]] = n_bins - 1          # include the upper edge in the last bin
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        counts[b] = n
        if n >= min_count:
            med[b] = 100.0 * np.median((est[sel] - true[sel]) / true[sel])
    return {"n": counts, "median_pct_err": med}


def plot_energy_dependence(true: np.ndarray, estimators: Dict[str, np.ndarray],
                           selections: Dict[str, np.ndarray], edges: Sequence[float],
                           out_file: str, *, target: str, epoch: float, range_label: str,
                           min_count: int = MIN_EVENTS_PER_BIN) -> List[Dict]:
    """
    Draw one figure for one target and one binning: a panel per event selection, each a 1D
    histogram of the median percent error vs true energy, one step line per estimator.

    true        : true values of the target, shape (n,)
    estimators  : label -> estimated values, shape (n,); the first entry is the model
    selections  : label -> boolean mask, shape (n,), e.g. {"all validation": ..., "passAnaFull": ...}
    out_file    : path the figure is written to (overwritten every epoch)
    range_label : name of the binning, e.g. "full" or "lowE" (goes into the CSV rows)

    Returns the binned numbers as a list of rows (for a CSV).
    """
    edges = np.asarray(edges, dtype=float)
    good = np.isfinite(true) & (true > 0)
    for v in estimators.values():
        good &= np.isfinite(v)

    n_cols = len(selections)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.8), sharey=True, squeeze=False)
    rows: List[Dict] = []

    for c, (sel_name, sel_mask) in enumerate(selections.items()):
        ax = axes[0, c]
        mask = good & sel_mask
        in_range = mask & (true >= edges[0]) & (true <= edges[-1])
        ax.axhline(0.0, color="#555555", lw=1.0, ls="--")

        for k, (label, est) in enumerate(estimators.items()):
            res = binned_median_pct_error(true[mask], est[mask], edges, min_count)
            ax.stairs(res["median_pct_err"], edges, color=_COLOURS[k % len(_COLOURS)],
                      ls=_LINESTYLES[k % len(_LINESTYLES)], lw=2.0, label=label,
                      baseline=None)
            for b in range(len(edges) - 1):
                rows.append({
                    "epoch": float(epoch), "target": target, "range": range_label,
                    "selection": sel_name, "estimator": label,
                    "bin_lo": edges[b], "bin_hi": edges[b + 1],
                    "n": int(res["n"][b]), "median_pct_err": res["median_pct_err"][b],
                })

        ax.set_title(f"{sel_name}  (N = {int(in_range.sum())} in range)", fontsize="medium")
        ax.set_xlabel(f"true {target} [GeV]")
        ax.set_xlim(edges[0], edges[-1])
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        if c == 0:
            ax.set_ylabel("median percent error: 100 x (est - true) / true [%]")
            ax.legend(fontsize="small")

    width = edges[1] - edges[0]
    fig.suptitle(
        f"Validation set, {target}: {edges[0]:g}-{edges[-1]:g} GeV, {width:g} GeV bins - "
        f"after epoch {epoch:.0f}\n(bins with fewer than {min_count} events are not drawn)",
        fontsize="medium",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return rows
