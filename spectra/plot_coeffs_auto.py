#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation des spectres d'auto-corrélation κκ & gg
  - Shot-noise retiré
  - Erreurs gaussiennes σ² = 2(C+SN)² / ((2l+1) f_sky)
  - Figures séparées pour galaxies et convergence du CMB
  - Ratio données/théorie inclus

Usage:
    python plot_coeffs_auto.py lrg_galaxy_auto.json pr4_kappa_auto.json fiducial/theory_bandpowers.json
"""

import argparse
import json
import pathlib
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# Constantes
SHOT = dict(LRGz1=4.01e-6, LRGz2=2.24e-6, LRGz3=2.08e-6, LRGz4=2.31e-6)
FSKY = 0.434
FULL_NAMES = {
    "LRGz1": "LRG bin z1",
    "LRGz2": "LRG bin z2",
    "LRGz3": "LRG bin z3",
    "LRGz4": "LRG bin z4",
    "kappa": "CMB Lensing (PR4)"
}
Z_RANGES = {
    "LRGz1": "0.4 < z < 0.6",
    "LRGz2": "0.6 < z < 0.8",
    "LRGz3": "0.8 < z < 1.0",
    "LRGz4": "1.0 < z < 1.2"
}
GALAXY_COLORS = {
    "LRGz1": "#663399",  # violet foncé
    "LRGz2": "#9370DB",  # violet moyen
    "LRGz3": "#9932CC",  # violet orchidée
    "LRGz4": "#8A2BE2",  # bleu violet
}
KAPPA_COLOR = "#FFD700"  # or

def read(fn):
    """Lecture des fichiers JSON contenant les spectres de puissance"""
    with open(fn) as f:
        d = json.load(f)
    ell = np.asarray(d["ell"])
    pat = re.compile(r"cl_([^_]+)_\1$")
    maps = [pat.match(k).group(1) for k in d if pat.match(k)]
    cls = {m: np.asarray(d[f"cl_{m}_{m}"]) for m in maps}
    return ell, cls

def classify_data(bundles):
    """Classifie les données en mesures et théorie"""
    data_files = []
    theory_file = None
    
    for label, ell, cls in bundles:
        if "theory" in label.lower() or "fiducial" in label.lower():
            theory_file = (label, ell, cls)
        else:
            data_files.append((label, ell, cls))
    
    return data_files, theory_file

def var_gauss(ell_local, cl, sn):
    """Calcul de la variance gaussienne"""
    return 2 * (cl + sn)**2 / ((2*ell_local + 1) * FSKY)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output", "-o", type=str, help="Préfixe pour les fichiers de sortie")
    args = ap.parse_args()

    # Lecture des données
    bundles = []
    for fn in args.json:
        ell_, cls_ = read(fn)
        label = pathlib.Path(fn).stem
        bundles.append((label, ell_, cls_))

    # Classification des données
    data_files, theory_file = classify_data(bundles)
    
    # Extraction des maps et séparation galaxies/kappa
    all_maps = sorted({m for _, _, c in bundles for m in c})
    galaxy_maps = [m for m in all_maps if m.startswith('LRG')]
    kappa_maps = [m for m in all_maps if m == 'PR4']
    
    # Base pour les noms de fichiers - SIMPLIFIÉE
    output_prefix = args.output if args.output else "spectra"
    
    # ==============================
    # FIGURE 1: SPECTRES DE GALAXIES
    # ==============================
    if galaxy_maps:
        fig_gal = plt.figure(figsize=(14, 8))
        gs_gal = GridSpec(2, len(galaxy_maps), height_ratios=[3, 1], hspace=0.05)
        
        # Création des axes
        axes_top = [fig_gal.add_subplot(gs_gal[0, i]) for i in range(len(galaxy_maps))]
        axes_bottom = [fig_gal.add_subplot(gs_gal[1, i], sharex=axes_top[i]) for i in range(len(galaxy_maps))]
        
        # Légende
        legend_handles = []
        legend_labels = []
        
        # Tracé pour chaque map de galaxies
        for i, m in enumerate(galaxy_maps):
            ax_top = axes_top[i]
            ax_bottom = axes_bottom[i]
            color = GALAXY_COLORS[m]
            
            # Tracé de la théorie
            if theory_file and m in theory_file[2]:
                theory_label, theory_ell, theory_cls = theory_file
                theory_sn = SHOT.get(m, 0.0)
                theory_spec = theory_cls[m] - theory_sn
                
                line, = ax_top.plot(theory_ell, theory_spec, 'k-', lw=2, alpha=0.7)
                if i == 0:
                    legend_handles.append(line)
                    legend_labels.append("Théorie (HEFT)")
            
            # Tracé des données mesurées
            for data_label, data_ell, data_cls in data_files:
                if m in data_cls:
                    sn = SHOT.get(m, 0.0)
                    spec = data_cls[m] - sn
                    err = np.sqrt(var_gauss(data_ell, spec, sn))
                    
                    eb = ax_top.errorbar(data_ell, spec, yerr=err, fmt='o', ms=6,
                                       capsize=3, color=color, alpha=0.9, elinewidth=1.5,
                                       markeredgecolor='black', markeredgewidth=0.5)
                    
                    if i == 0:
                        legend_handles.append(eb)
                        legend_labels.append("Galaxies LRG")
                    
                    # Ratio données/théorie
                    if theory_file and m in theory_file[2]:
                        theory_ell = theory_file[1]
                        theory_spec = theory_file[2][m] - SHOT.get(m, 0.0)
                        
                        theory_interp = interp1d(theory_ell, theory_spec,
                                              bounds_error=False,
                                              fill_value=(theory_spec[0], theory_spec[-1]))
                        theory_at_data = theory_interp(data_ell)
                        
                        ratio = spec / theory_at_data
                        ratio_err = err / theory_at_data
                        
                        ax_bottom.errorbar(data_ell, ratio, yerr=ratio_err,
                                         fmt='o', ms=6, capsize=3, color=color, alpha=0.9,
                                         elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)
            
            # Finalisation du panneau supérieur
            map_title = FULL_NAMES.get(m, m)
            if m in Z_RANGES:
                map_title += f" ({Z_RANGES[m]})"
            
            ax_top.set_title(map_title, fontsize=12, pad=10)
            ax_top.set_yscale('log')
            ax_top.set_xscale('log')
            
            if i == 0:
                ax_top.set_ylabel(r"$C_\ell$ (shot-noise retiré)", fontsize=11)
            
            # Limites des axes adaptées à chaque bin
            if m == 'LRGz1': ax_top.set_ylim(1e-6, 3e-4)
            elif m == 'LRGz2': ax_top.set_ylim(5e-7, 1e-4)
            elif m == 'LRGz3': ax_top.set_ylim(5e-7, 1e-4)
            elif m == 'LRGz4': ax_top.set_ylim(1e-7, 5e-5)
            
            # Panneau de ratio
            ax_top.tick_params(labelbottom=False)
            ax_bottom.axhline(1.0, color='k', ls='--', alpha=0.5)
            ax_bottom.set_ylim(0.5, 1.5)
            ax_bottom.set_xlabel(r"$\ell$", fontsize=11)
            
            if i == 0:
                ax_bottom.set_ylabel(r"Données/Théorie", fontsize=11)
            
            ax_bottom.set_xscale('log')
            ax_bottom.set_xlim(30, 3000)
        
        # Légende globale
        fig_gal.legend(legend_handles, legend_labels,
                    loc='upper center', bbox_to_anchor=(0.5, 0.98),
                    ncol=len(legend_labels), frameon=False)
        
        fig_gal.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarde avec nom SIMPLIFIÉ
        gal_output = f"{output_prefix}_galaxy"
        fig_gal.savefig(gal_output + ".pdf")
        fig_gal.savefig(gal_output + ".png")
        print(f"→ Galaxies : {gal_output}.pdf et {gal_output}.png")
    
    # ==============================
    # FIGURE 2: SPECTRE KAPPA-KAPPA
    # ==============================
    if kappa_maps:
        fig_kappa = plt.figure(figsize=(8, 8))
        gs_kappa = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        
        ax_top = fig_kappa.add_subplot(gs_kappa[0, 0])
        ax_bottom = fig_kappa.add_subplot(gs_kappa[1, 0], sharex=ax_top)
        
        legend_handles = []
        legend_labels = []
        
        # Théorie pour kappa
        if theory_file and 'PR4' in theory_file[2]:
            theory_label, theory_ell, theory_cls = theory_file
            theory_spec = theory_cls['PR4']  # pas de shot-noise pour kappa
            
            line, = ax_top.plot(theory_ell, theory_spec, 'k-', lw=2, alpha=0.7)
            legend_handles.append(line)
            legend_labels.append("Théorie (HEFT)")
        
        # Données kappa
        for data_label, data_ell, data_cls in data_files:
            if 'PR4' in data_cls:
                spec = data_cls['PR4']
                err = np.sqrt(var_gauss(data_ell, spec, 0.0))
                
                eb = ax_top.errorbar(data_ell, spec, yerr=err, fmt='o', ms=6,
                                   capsize=3, color=KAPPA_COLOR, alpha=0.9, elinewidth=1.5,
                                   markeredgecolor='black', markeredgewidth=0.5)
                
                legend_handles.append(eb)
                legend_labels.append("CMB Lensing (PR4)")
                
                # Ratio données/théorie
                if theory_file and 'PR4' in theory_file[2]:
                    theory_ell = theory_file[1]
                    theory_spec = theory_file[2]['PR4']
                    
                    theory_interp = interp1d(theory_ell, theory_spec,
                                          bounds_error=False,
                                          fill_value=(theory_spec[0], theory_spec[-1]))
                    theory_at_data = theory_interp(data_ell)
                    
                    ratio = spec / theory_at_data
                    ratio_err = err / theory_at_data
                    
                    ax_bottom.errorbar(data_ell, ratio, yerr=ratio_err,
                                     fmt='o', ms=6, capsize=3, color=KAPPA_COLOR, alpha=0.9,
                                     elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)
        
        # Finalisation du panneau supérieur pour kappa
        ax_top.set_title(r"CMB Lensing (PR4) - $C_\ell^{\kappa\kappa} + N_\ell^{\kappa\kappa}$", fontsize=12, pad=10)
        ax_top.set_yscale('log')
        ax_top.set_xscale('log')
        ax_top.set_ylabel(r"$C_\ell^{\kappa\kappa}$", fontsize=12)
        ax_top.set_ylim(1e-8, 1e-5)
        ax_top.tick_params(labelbottom=False)
        
        # Finalisation du panneau inférieur
        ax_bottom.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax_bottom.set_ylim(0.5, 1.5)
        ax_bottom.set_xlabel(r"$\ell$", fontsize=12)
        ax_bottom.set_ylabel(r"Données/Théorie", fontsize=12)
        ax_bottom.set_xscale('log')
        ax_bottom.set_xlim(30, 3000)
        
        # Légende
        fig_kappa.legend(legend_handles, legend_labels,
                      loc='upper center', bbox_to_anchor=(0.5, 0.98),
                      ncol=len(legend_labels), frameon=False)
        
        fig_kappa.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarde
        kappa_output = f"{output_prefix}_kappa_PR4"
        fig_kappa.savefig(kappa_output + ".pdf")
        fig_kappa.savefig(kappa_output + ".png")
        print(f"→ Kappa : {kappa_output}.pdf et {kappa_output}.png")
    
    if args.show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()