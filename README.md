[![](https://img.shields.io/badge/arXiv-2106.09713%20-red.svg)](https://arxiv.org/abs/24MM.XXXXX)

# Maps to Parameters

Code used for the cross-correlation analysis of DESI Luminous Red Galaxies with CMB lensing from Planck (PR3 and PR4) and ACT DR6.
- making the angular power spectrum measurements: `spectra`
- the likelihood: `likelihood/`,`theory/`,`yamls/`

# Installation

To install `MaPar` run `bash setup.sh`. Doing so will install all required dependencies:

`numpy`, `scipy`, `CLASS`, `velocileptors`, ...

[`CLASS`](https://github.com/lesgourg/class_public) is used to compute CMB anisotropies and the linear matter power spectrum. 

[`velocileptors`](https://github.com/sfschen/velocileptors) is used to compute the the redshift-space power spectrum of a general biased tracer to 1-loop order in Lagrangian Perturbation Theory. 


# Code Structure

still in flux

# Basic Usage

need to write the code first

`plot_chains.py` accepts a `--label` option to set the legend label used in
triangle plots. If omitted, the plot label defaults to the chain's file root.

The `maps` directory provides small utilities for visualizing the data. For
instance `plot_kappa_maps.py` displays the Planck PR4 convergence map, and
`plot_lrg_maps.py` shows the DESI LRG galaxy density maps.
