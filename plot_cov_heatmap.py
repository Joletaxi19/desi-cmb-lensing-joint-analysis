#!/usr/bin/env python3
"""Display a correlation matrix heatmap from a covariance file."""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage


def load_covariance(path):
    """Return parameter names and covariance matrix from ``path``."""

    with open(path) as f:
        header = f.readline().lstrip("#").split()
    cov = np.loadtxt(path, skiprows=1)
    return header, cov


def covariance_to_correlation(cov):
    """Convert a covariance matrix to a correlation matrix."""

    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / std[:, None] / std[None, :]
    corr[np.isnan(corr)] = 0
    return corr


def reorder_params(names, corr):
    """Group parameters (cosmo, b1, b2, bs, smag) and apply clustering."""

    def category(n):
        for key in ["b1_", "b2_", "bs_", "smag_"]:
            if n.startswith(key):
                return key[:-1]
        return "cosmo"

    groups = {"cosmo": [], "b1": [], "b2": [], "bs": [], "smag": []}
    for i, n in enumerate(names):
        groups[category(n)].append(i)

    order = []
    for g in ["cosmo", "b1", "b2", "bs", "smag"]:
        if len(groups[g]) > 1:
            # Hierarchical clustering within group
            sub_corr = corr[np.ix_(groups[g], groups[g])]
            d = 1 - sub_corr
            linkage_mat = linkage(d[np.tril_indices_from(d, k=-1)], method="average")
            order.extend([groups[g][i] for i in leaves_list(linkage_mat)])
        else:
            order.extend(groups[g])

    reordered_names = [names[i] for i in order]
    reordered_corr = corr[np.ix_(order, order)]
    return reordered_names, reordered_corr


def clean_label(name):
    """Shorten parameter labels for display."""

    name = name.replace("LRG", "")
    return name


def main():
    ap = argparse.ArgumentParser(description="Trace la matrice de corrélation")
    ap.add_argument(
        "--covmat",
        default="yamls/chains/my_pr4.covmat",
        help="fichier de covariance",
    )
    ap.add_argument("--output", default="correlation_heatmap.png", help="image de sortie")
    args = ap.parse_args()

    params, cov = load_covariance(args.covmat)
    corr = covariance_to_correlation(cov)
    params, corr = reorder_params(params, corr)

    display_labels = [clean_label(p) for p in params]

    mask = np.triu(np.ones_like(corr, dtype=bool))
    annot = np.empty_like(corr, dtype=object)
    annot[:] = ""
    for i in range(corr.shape[0]):
        for j in range(i + 1):
            if abs(corr[i, j]) >= 0.8:
                annot[i, j] = f"{corr[i, j]:.2f}"

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Correlation coefficient"},
        annot=annot,
        fmt="s",
        annot_kws={"size": 8},
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    ax.set_title("Matrice de corrélation des paramètres")
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
