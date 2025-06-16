#!/usr/bin/env python3
"""Display a correlation matrix heatmap for power spectra.

Given a JSON file produced by the power-spectrum pipeline (for example
``lrg_galaxy_auto.json`` or ``lrg_cross_pr4.json``), this script
extracts the covariance matrix for a chosen pair of maps and displays
its correlation matrix as a heatmap.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_json(path):
    """Return ell values and covariance matrices from ``path``."""
    with open(path) as f:
        data = json.load(f)
    ell = np.asarray(data.get("ell") or data.get("ells"))
    covs = {
        k[4:]: np.asarray(v) for k, v in data.items() if k.startswith("cov_")
    }
    return ell, covs


def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / std[:, None] / std[None, :]
    corr[np.isnan(corr)] = 0
    return corr


def main():
    ap = argparse.ArgumentParser(description="Affiche une matrice de corrélation")
    ap.add_argument("json", help="fichier JSON contenant les C_ell")
    ap.add_argument("pair", help="nom du spectre, ex: PR4_LRGz1")
    ap.add_argument("--output", "-o", default=None, help="image de sortie")
    args = ap.parse_args()

    ell, covs = load_json(args.json)
    key = args.pair if args.pair.startswith("cov_") else args.pair
    key = key.replace("cov_", "")
    if key not in covs:
        raise ValueError(f"Covariance for {key} not found in {args.json}")

    corr = covariance_to_correlation(covs[key])

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"label": "coefficient"},
        ax=ax,
    )
    ax.set_title(f"Correlation matrix: {key}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell$")
    plt.tight_layout()

    out = args.output or f"corr_{Path(args.json).stem}_{key}.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()

