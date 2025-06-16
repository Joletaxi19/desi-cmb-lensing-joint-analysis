#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the cross spectra :math:`C_{\ell}^{\kappa g}`.

Several JSON files can be given.  If one of them contains the word
``theory`` or ``fiducial`` in its file name, it is interpreted as the
theoretical prediction and plotted as a line.  All the other files are
considered data and are plotted with error bars.  A ratio data/theory
panel is shown underneath each spectrum, similarly to
``plot_coeffs_auto.py``.

Examples
--------
```
python plot_coeffs_croisees.py lrg_cross_pr4.json fiducial/theory_bandpowers.json
python plot_coeffs_croisees.py lrg_cross_pr3.json lrg_cross_pr4.json
```
"""

import json
import argparse
import re
import itertools
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# ---------- STYLE ----------
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 140,
})
COLORS   = ["#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#CC6677", "#882255"]
MARKERS  = ["o", "s", "D", "^", "v", ">", "<", "P"]

FULL_NAMES = {
    "LRGz1": "LRG bin z1",
    "LRGz2": "LRG bin z2",
    "LRGz3": "LRG bin z3",
    "LRGz4": "LRG bin z4",
}

# ---------- OUTILS ----------
def load_json(fn):
    """Return ``ell``, spectra and covariances for the file ``fn``."""

    with open(fn) as f:
        d = json.load(f)

    ell = np.asarray(d.get("ell") or d.get("ells"))

    tag_pat = re.compile(r"cl_([^_]+)_(.+)")
    tag = next(tag_pat.match(k).group(1) for k in d if k.startswith("cl_"))
    bin_pat = re.compile(rf"cl_{re.escape(tag)}_(.+)")
    bins = sorted({bin_pat.match(k).group(1) for k in d if bin_pat.match(k)})

    cls = {b: np.asarray(d[f"cl_{tag}_{b}"]) for b in bins}
    covs = {
        b: np.asarray(d.get(f"cov_{tag}_{b}_{tag}_{b}", np.zeros((len(ell), len(ell)))) )
        for b in bins
    }

    return pathlib.Path(fn).stem, ell, cls, covs, bins


def classify_data(bundles):
    """Separate data files and (optional) theory file."""

    data = []
    theory = None

    for label, ell, cls, covs, bins in bundles:
        if "theory" in label.lower() or "fiducial" in label.lower():
            theory = (label, ell, cls)
        else:
            data.append((label, ell, cls, covs))

    return data, theory


def make_panels(bins, data_files, theory_file, output_prefix):
    """Plot cross-spectra with optional theory overlay."""

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, len(bins), height_ratios=[3, 1], hspace=0.05)

    axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(bins))]
    axes_bot = [fig.add_subplot(gs[1, i], sharex=axes_top[i]) for i in range(len(bins))]

    legend_handles = []
    legend_labels = []

    for i, b in enumerate(bins):
        ax = axes_top[i]
        ax_r = axes_bot[i]

        # Theory line
        if theory_file and b in theory_file[2]:
            th_ell = theory_file[1]
            th_spec = theory_file[2][b]
            line, = ax.plot(th_ell, th_spec, 'k-', lw=2, alpha=0.7)
            if i == 0:
                legend_handles.append(line)
                legend_labels.append("Théorie")

        for (label, ell, cls, covs), color, marker in zip(data_files, itertools.cycle(COLORS), itertools.cycle(MARKERS)):
            spec = cls[b]
            err = np.sqrt(np.diag(covs[b]))
            eb = ax.errorbar(ell, spec, yerr=err, fmt=marker, ms=5, capsize=2,
                             color=color, label=label)

            if i == 0:
                legend_handles.append(eb)
                legend_labels.append(label)

            if theory_file and b in theory_file[2]:
                th_interp = interp1d(th_ell, th_spec, bounds_error=False,
                                     fill_value=(th_spec[0], th_spec[-1]))
                th_at_data = th_interp(ell)
                ratio = spec / th_at_data
                ratio_err = err / th_at_data
                ax_r.errorbar(ell, ratio, yerr=ratio_err, fmt=marker, ms=5,
                              capsize=2, color=color)

        title = FULL_NAMES.get(b, b)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(ls=':')

        ax_r.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax_r.set_ylim(0.5, 1.5)
        ax_r.set_xscale('log')
        ax_r.set_xlabel(r"$\ell$")
        ax_r.grid(ls=':')
        if i == 0:
            ax.set_ylabel(r"$C_\ell^{\kappa g}$")
            ax_r.set_ylabel("Données/Th")

        ax_r.set_xlim(30, 3000)

    fig.legend(legend_handles, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_prefix + ".png")
    fig.savefig(output_prefix + ".pdf")
    print(f"→ {output_prefix}.pdf et {output_prefix}.png")

# ---------- MAIN ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", help="fichiers lrg_cross_*.json ou theory")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output", "-o", default="cross", help="préfixe de sortie")
    args = ap.parse_args()

    bundles = [load_json(fn) for fn in args.json]

    data_files, theory_file = classify_data(bundles)
    bins = bundles[0][4]

    make_panels(bins, data_files, theory_file, args.output)

    if args.show:
        plt.show()
    else:
        plt.close("all")
