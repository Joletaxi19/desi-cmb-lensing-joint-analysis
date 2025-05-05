# lrg_galaxy_auto.py
#
# Calcule les auto-spectres (gg) de chaque échantillon LRG (z1–z4)
# ainsi que toutes les matrices de covariance associées.  Les classes
# croisées entre échantillons ne sont pas calculées (only_auto=True).

import sys, healpy as hp
from calc_cl import full_master
sys.path.append('../')

from globe import LEDGES, NSIDE

# --- Lecture des maps & masques --------------------------------------------
lrg_maps  = [hp.read_map(f'../maps/lrg_s0{isamp}_del.hpx2048.public.fits.gz')
             for isamp in range(1, 5)]
lrg_masks = [hp.read_map(f'../maps/masks/lrg_s0{isamp}_msk.hpx2048.public.fits.gz')
             for isamp in range(1, 5)]

# --- Appel générique --------------------------------------------------------
galNames = ['LRGz1', 'LRGz2', 'LRGz3', 'LRGz4']
fnout    = 'lrg_galaxy_auto.json'

full_master(
    LEDGES,
    maps       = lrg_maps,
    msks       = lrg_masks,
    names      = galNames,
    fnout      = fnout,
    do_cov     = True,          # matrice de covariance gaussienne
    only_auto  = True,          # on ne veut que les auto-spectres
    pairs      = [[i, i] for i in range(4)]
    # on ne fournit pas cij : il sera ajusté par polynôme aux gg mesurés
)

print(f'Script terminé : résultats dans {fnout}')