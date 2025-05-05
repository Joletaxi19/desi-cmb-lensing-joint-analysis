#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace autos κκ & gg
  – shot-noise retiré,
  – erreurs gaussiennes σ² = 2(C+SN)² / ((2ℓ+1) f_sky).

Usage :
    python plot_coeffs_auto.py lrg_galaxy_auto.json theory_bandpowers.json
"""

import argparse, json, pathlib, itertools, math, re
import numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({"font.size":10,"legend.fontsize":8,
                     "xtick.direction":"in","ytick.direction":"in","figure.dpi":140})
COL  = ["#332288","#88CCEE","#44AA99","#117733","#CC6677","#882255"]
MAR  = ["o","s","D","^","v","P"]

SHOT = dict(LRGz1=4.01e-6, LRGz2=2.24e-6, LRGz3=2.08e-6, LRGz4=2.31e-6)
FSKY = 0.434         

def read(fn):
    with open(fn) as f:
        d = json.load(f)
    ell = np.asarray(d["ell"])
    pat = re.compile(r"cl_([^_]+)_\1$")
    maps = [pat.match(k).group(1) for k in d if pat.match(k)]
    cls  = {m: np.asarray(d[f"cl_{m}_{m}"]) for m in maps}
    return ell, cls

def grid(n):
    ncol = math.ceil(math.sqrt(n))
    nrow = math.ceil(n / ncol)
    return nrow, ncol

def var_gauss(ell_local, cl, sn):
    return 2 * (cl + sn)**2 / ((2*ell_local + 1) * FSKY)

# ---------- main ----------
ap = argparse.ArgumentParser()
ap.add_argument("json", nargs="+")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

bundles = []
for fn in args.json:
    ell_, cls_ = read(fn)
    label = pathlib.Path(fn).stem
    bundles.append((label, ell_, cls_))

all_maps = sorted({m for _,_,c in bundles for m in c})
nrow, ncol = grid(len(all_maps))

fig, axarr = plt.subplots(nrow, ncol,
                          figsize=(3.4*ncol, 2.8*nrow),
                          sharex=True)
axarr = np.atleast_1d(axarr).ravel()

for i, m in enumerate(all_maps):
    ax = axarr[i]
    marker = MAR[i % len(MAR)]
    for col, (lab, ell_b, cls_b) in zip(itertools.cycle(COL), bundles):
        if m not in cls_b:
            continue
        sn   = SHOT.get(m, 0.0)
        spec = cls_b[m] - sn
        err  = np.sqrt(var_gauss(ell_b, spec, sn))
        ax.errorbar(ell_b, spec, yerr=err,
                    fmt=marker+"-", ms=4, lw=1, capsize=2,
                    color=col, label=lab)
    ax.set_yscale("log"); ax.set_title(m); ax.grid(ls=":")
    if i//ncol == nrow-1:
        ax.set_xlabel(r"$\ell$")
    if i % ncol == 0:
        ax.set_ylabel(r"$C_\ell$  (shot-noise retiré)")

for j in range(len(all_maps), nrow*ncol):
    fig.delaxes(axarr[j])

handles, labels = axarr[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False,
           loc="upper center", ncol=min(4, len(labels)))
fig.tight_layout(rect=[0,0,1,0.94])

out = "_".join(p for p,_,_ in bundles)
fig.savefig(out)
print("→", out)

if args.show:
    plt.show()
else:
    plt.close("all")