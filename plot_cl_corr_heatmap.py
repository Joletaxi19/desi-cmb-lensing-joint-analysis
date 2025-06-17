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
from likelihoods.pack_data import get_cov


def load_json(path):
    """Return ell values and covariance matrices from ``path``."""
    with open(path) as f:
        data = json.load(f)
    ell = np.asarray(data.get("ell") or data.get("ells"))
    covs = {
        k[4:]: np.asarray(v) for k, v in data.items() if k.startswith("cov_")
    }
    return ell, covs


def load_merged_json(cross_path, auto_path):
    """Return dictionary containing all covariance matrices from two files."""
    ell1, cov1 = load_json(cross_path)
    ell2, cov2 = load_json(auto_path)
    if len(ell1) != len(ell2) or not np.allclose(ell1, ell2):
        raise ValueError("Ell arrays do not match between input files")
    data = {"ell": ell1}
    for k, v in cov1.items():
        data[f"cov_{k}"] = v
    for k, v in cov2.items():
        data[f"cov_{k}"] = v
    return data


def build_full_covariance(data, kap_name="PR4", gal_names=None):
    """Concatenate all covariance sub-blocks into a single matrix."""
    if gal_names is None:
        gal_names = [f"LRGz{i}" for i in range(1, 5)]

    ell = np.asarray(data.get("ell") or data.get("ells"))
    nell = len(ell)

    pairs = [(g, g) for g in gal_names] + [
        (kap_name, g) for g in gal_names
    ] + [(kap_name, kap_name)]

    n = len(pairs)
    cov = np.zeros((n * nell, n * nell))
    for i, (a1, a2) in enumerate(pairs):
        for j, (b1, b2) in enumerate(pairs):
            cov_block = get_cov(data, a1, a2, b1, b2)
            cov[i * nell : (i + 1) * nell, j * nell : (j + 1) * nell] = cov_block

    labels = [f"{p[0]}_{p[1]}" for p in pairs]
    return ell, cov, labels


def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / std[:, None] / std[None, :]
    corr[np.isnan(corr)] = 0
    return corr


def main():
    ap = argparse.ArgumentParser(
        description="Affiche une matrice de corrélation pour un spectre ou l'ensemble des observables"
    )
    ap.add_argument("json", nargs="?", help="fichier JSON contenant les C_ell")
    ap.add_argument("pair", nargs="?", help="nom du spectre, ex: PR4_LRGz1")
    ap.add_argument("--cross-json", default="spectra/lrg_cross_pr4.json", help="fichier de covariance croisée")
    ap.add_argument("--auto-json", default="spectra/pr4_kappa_auto.json", help="fichier de covariance auto κκ")
    ap.add_argument("--full", action="store_true", help="affiche la matrice complète")
    ap.add_argument("--output", "-o", default=None, help="image de sortie")
    args = ap.parse_args()

    if args.full:
        data = load_merged_json(args.cross_json, args.auto_json)
        ell, cov, labels = build_full_covariance(data)
        corr = covariance_to_correlation(cov)

        sns.set_context("talk")
        fig, ax = plt.subplots(figsize=(8, 6))
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
        ax.set_title("Correlation matrix: all spectra")
        ax.set_xlabel("data vector index")
        ax.set_ylabel("data vector index")
        plt.tight_layout()
        out = args.output or "corr_all_spectra.png"
        plt.savefig(out)
        plt.close(fig)
        print(f"→ {out}")
        return

    if args.json is None or args.pair is None:
        ap.error("json et pair requis sans --full")

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

