#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertit cls_LRGxPR4_bestFit.txt en pseudo-Cℓ binés (« band-powers »)
compatibles avec les bin edges LEDGES & le pixwin des cartes 2048.

  * sort un fichier theory_bandpowers.json
  * les clés sont cl_PR4_PR4 (κκ), cl_PR4_LRGz1 …, cl_LRGz1_LRGz1, etc.
  * chaque auto gg est stocké 2× :   ...      (avec shot-noise)
                                     ..._noSN (shot-noise retiré)

"""

import json, numpy as np, healpy as hp, pymaster as nmt
import sys
sys.path.append("../..")  # pour globe.py
from globe import LEDGES, NSIDE

TXT = "cls_LRGxPR4_bestFit.txt"
OUT = "theory_bandpowers.json"
ELL_MAX = 3*NSIDE
names = ["PR4", "LRGz1", "LRGz2", "LRGz3", "LRGz4"]
SN    = np.array([0, 4.01e-6, 2.24e-6, 2.08e-6, 2.31e-6])  # plateau  shot

# --- charge les Cℓ (déjà filtrés & pixwin inclus) --------------------------
cij = np.loadtxt(TXT).reshape((5,5,ELL_MAX))

# charge le bruit N_ell^kk filtré PR4 et ajoute au C_ell
nkk = np.loadtxt("../../data/PR4_lens_nlkk_filt.txt")
Nvec = np.interp(np.arange(ELL_MAX), nkk[:,0], nkk[:,2], right=0)

cij[0,0] += Nvec

# --- convertit en band-powers MASTER ---------------------------------------
bins   = nmt.NmtBin.from_edges(np.array(LEDGES[:-1]), np.array(LEDGES[1:]))
# --- matrice top-hat : un bandeau = 1 à l'intérieur des bornes ------------
n_bands = len(LEDGES) - 1          # une bande entre chaque paire d'arêtes
w = np.zeros((n_bands, ELL_MAX))

for b, (l0, l1) in enumerate(zip(LEDGES[:-1], LEDGES[1:])):
    w[b, l0:l1] = 1.0              # poids uniformes dans la bande

bp_cij = np.einsum("bl,ij l -> ij b", w, cij) / (np.array(LEDGES[1:]) - np.array(LEDGES[:-1]))

# --- prépare la sortie ------------------------------------------------------
out = {"ell": list(0.5*(np.array(LEDGES[:-1])+LEDGES[1:])),
       "map names": names, "nside": NSIDE}

for i in range(5):
    for j in range(i,5):
        key = f"cl_{names[i]}_{names[j]}"
        out[key] = bp_cij[i,j].tolist()
        # version sans shot-noise pour les autos gg :
        if i==j and i>0:
            out[key+"_noSN"] = (bp_cij[i,i] - SN[i]).tolist()

# covariances → placeholders nuls
M = len(out["ell"])
zero_cov = np.zeros((M,M)).tolist()
for i in range(5):
    key = f"cov_{names[i]}_{names[i]}_{names[i]}_{names[i]}"
    out[key] = zero_cov

with open(OUT, "w") as f: json.dump(out, f, indent=2)
print("→ théorie binaire écrite dans", OUT)