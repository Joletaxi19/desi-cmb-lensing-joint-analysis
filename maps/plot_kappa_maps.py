import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ------------------------------------------------------------------
# 0) Fichiers d’entrée ------------------------------------------------
kap_file  = "PR4_lens_kap_filt.hpx2048.fits"
mask_file = "masks/PR4_lens_mask.fits"

# ------------------------------------------------------------------
# 1) Lecture des cartes ---------------------------------------------
kappa = hp.read_map(kap_file)
mask  = hp.read_map(mask_file)

# ------------------------------------------------------------------
# 2) Lissage puis masquage ------------------------------
good = mask > 1 - 1e-1
bad  = ~good

fwhm_deg = 1.0 # lissage en degrés
kappa_sm = hp.smoothing(
    kappa,                    # lisser la carte complète
    fwhm=np.radians(fwhm_deg),
)

# Appliquer le masque net après le lissage
kappa_sm[bad] = hp.UNSEEN

# ------------------------------------------------------------------
# 4) Colormap -----------------------------------------------

cmap = plt.get_cmap('YlGnBu')
cmap.set_bad("gray")
cmap.set_under("white")

vmin, vmax = -0.05, 0.05

# ------------------------------------------------------------------
# 5) Figure ----------------------------------------------------------
fig = plt.figure(figsize=(14, 7))

hp.orthview(
    kappa_sm,
    rot=(0, 0, 0),           # centre galactique
    half_sky=True,
    sub=(1, 2, 1),
    coord="G",
    cmap=cmap,
    min=vmin, max=vmax,
    notext=True, cbar=False,
    title="",
    xsize=7000,
)

hp.orthview(
    kappa_sm,
    rot=(180, 0, 0),         # anti-centre
    half_sky=True,
    sub=(1, 2, 2),
    coord="G",
    cmap=cmap,
    min=vmin, max=vmax,
    notext=True, cbar=False,
    title="",
    xsize=7000,
)

# 5) colorbar
import matplotlib as mpl
cax = fig.add_axes([0.10, 0.0, 0.80, 0.02])
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm   = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.ax.set_xlabel(r"$\kappa$", fontsize=12)

# 6) annotation en haut (légende)
if fwhm_deg != 0.0:
    fig.text(0.5, 1,
            f"Carte de convergence Planck PR4 – $\\ell<2500$, lissée à {fwhm_deg}° FWHM",
            ha="center", va="top", fontsize=14)
else:
    fig.text(0.5, 1,
            f"Carte de convergence Planck PR4 – $\\ell<2500$",
            ha="center", va="top", fontsize=14)

# 7) sauvegarde
plt.savefig(
    "planck_pr4_kappa.png",
    dpi=300,
    bbox_inches="tight"
)