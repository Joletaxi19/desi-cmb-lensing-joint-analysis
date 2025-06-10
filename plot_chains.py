#!/usr/bin/env python3
"""Visualise Cobaya MCMC chains using helper functions.

This script loads chains produced by Cobaya and stored in ``yamls/chains``.
It relies on the plotting utilities defined in ``notebooks/viewchains.py``
to generate simple triangle plots of the parameters of interest.
"""

import argparse
import os
import numpy as np
from getdist.mcsamples import loadMCSamples

from notebooks.viewchains import add_S8x, get_bestFit_values, make_triangle_plot


def main():
    ap = argparse.ArgumentParser(description="Plot a chain using getdist")
    ap.add_argument("--chain_dir", default="yamls/chains", help="directory with chain files")
    ap.add_argument("--file_root", default="my_pr4", help="prefix of the chain files")
    ap.add_argument(
        "--label",
        default=None,
        help="legend label (default: use file_root)",
    )
    ap.add_argument(
        "--params", nargs="+", default=["OmM", "sigma8", "S8x"],
        help="parameters to display in the triangle plot",
    )
    ap.add_argument("--output", "-o", help="output image filename")
    args = ap.parse_args()

    legend_label = args.label if args.label is not None else args.file_root

    root = os.path.join(args.chain_dir, args.file_root)
    chain = loadMCSamples(root)

    param_names = [p.name for p in chain.getParamNames().names]
    if "S8x" not in param_names:
        add_S8x(chain)

    maxima = None
    try:
        bf = get_bestFit_values(root)
        maxima = np.array([[bf[p] for p in args.params]])
    except Exception:
        pass

    make_triangle_plot(
        [chain],
        [legend_label],
        args.params,
        filled=True,
        save_path=args.output,
        maxima=maxima,
    )


if __name__ == "__main__":
    main()
