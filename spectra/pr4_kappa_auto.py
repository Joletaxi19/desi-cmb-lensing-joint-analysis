# pr4_kappa_auto.py
#
# Calcule le spectre de puissance auto-corrélé (κκ) de la carte
# de lentille CMB Planck PR4, ainsi que sa matrice de covariance
# gaussienne, et les enregistre sous forme JSON.

import sys, numpy as np, healpy as hp
from calc_cl import full_master
sys.path.extend(['../', '../mc_correction/'])

from globe import LEDGES, NSIDE

# --- Lecture des données ----------------------------------------------------
kap_map  = [hp.read_map('../maps/PR4_lens_kap_filt.hpx2048.fits')]
kap_mask = [hp.read_map('../maps/masks/PR4_lens_mask.fits')]
kap_nkk  = np.loadtxt('../data/PR4_lens_nlkk_filt.txt')     # N_ℓ^{κκ}

# --- Préparation des « théories » pour la covariance ------------------------
ells = np.arange(3*2048)                 # 3·nside = ℓ_max utilisable
cij  = np.zeros((1, 1, len(ells)))
cij[0, 0] = np.interp(ells, kap_nkk[:, 0], kap_nkk[:, 2], right=0)

# --- Appel générique --------------------------------------------------------
fnout = 'pr4_kappa_auto.json'
names = ['PR4']

full_master(
    LEDGES,
    maps   = kap_map,
    msks   = kap_mask,
    names  = names,
    fnout  = fnout,
    cij    = cij,
    do_cov = True,
    pairs  = [[0, 0]]                    # juste κκ
)

print(f'Script terminé : résultats dans {fnout}')