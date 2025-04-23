#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage :
    python plot_cl.py lrg_cross_pr4.json
    python plot_cl.py lrg_cross_pr3.json lrg_cross_pr4.json
"""
import json, argparse, re, itertools, pathlib
import numpy as np
import matplotlib.pyplot as plt

# ---------- STYLE ----------
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 140,
})
COLORS   = ["#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#CC6677", "#882255"]
MARKERS  = ["o", "s", "D", "^", "v", ">", "<", "P"]

# ---------- OUTILS ----------
def load_json(fn):
    """-> ell, {bin:cl}, {bin:cov}, TAG."""
    with open(fn) as f:
        d = json.load(f)
    ell = np.asarray(d.get("ell") or d.get("ells"))
    # trouve TAG et bins
    tag_pat = re.compile(r"cl_([^_]+)_(.+)")
    tag = next(tag_pat.match(k).group(1) for k in d if k.startswith("cl_"))
    bin_pat = re.compile(rf"cl_{re.escape(tag)}_(.+)")
    bins = sorted({bin_pat.match(k).group(1) for k in d if bin_pat.match(k)})
    cls  = {b: np.asarray(d[f"cl_{tag}_{b}"]) for b in bins}
    covs = {b: np.asarray(d[f"cov_{tag}_{b}_{tag}_{b}"]) for b in bins}
    return ell, cls, covs, tag.upper(), bins

def make_overview(ell, bundles, outfile):
    """toutes les releases sur un seul axe."""
    plt.figure(figsize=(6,4))
    multi = len(bundles) > 1
    for c,(tag,cls,covs,bins) in zip(itertools.cycle(COLORS), bundles):
        for m,b in zip(MARKERS, bins):
            err = np.sqrt(np.diag(covs[b]))
            lab = f"{tag} {b}" if multi else b
            plt.errorbar(ell, cls[b], yerr=err,
                         fmt=m+"-", ms=4, lw=1, capsize=2,
                         color=c if multi else None, label=lab)
    plt.yscale("log"); plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell^{\kappa g}$")
    plt.title("Spectres croisés κ×LRG"); plt.grid(ls=":")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout(); plt.savefig(outfile)

def make_panels(ell, bundles, outfile):
    """un panel par bin, releases superposées."""
    bins = bundles[0][3]                     # même liste pour tous
    fig, axes = plt.subplots(2,2, figsize=(7,5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax,b in zip(axes, bins):
        for c,(tag,cls,covs,_) in zip(COLORS, bundles):
            m = MARKERS[list(bins).index(b)]
            err = np.sqrt(np.diag(covs[b]))
            ax.errorbar(ell, cls[b], yerr=err,
                        fmt=m, ms=4, lw=1, capsize=2,
                        color=c, label=tag)
        ax.set_yscale("log"); ax.set_title(b.replace("LRG","LRG "))
        ax.grid(ls=":")
    axes[2].set_xlabel(r"$\ell$"); axes[3].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$C_\ell$"); axes[2].set_ylabel(r"$C_\ell$")
    # légende commune
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(bundles)], labels[:len(bundles)],
               frameon=False, loc="upper center", ncol=len(bundles))
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(outfile)

# ---------- MAIN ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", help="fichiers lrg_cross_*.json")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bundles = []
    for fn in args.json:
        ell, cls, covs, tag, bins = load_json(fn)
        bundles.append((tag, cls, covs, bins))

    suf = "_".join([b[0] for b in bundles])
    make_overview(ell, bundles, f"{suf}_overview.png")
    make_panels  (ell, bundles, f"{suf}_panels.png")

    if args.show: plt.show()
    else:
        plt.close("all")
        print(f"→ {suf}_overview.png  &  {suf}_panels.png")