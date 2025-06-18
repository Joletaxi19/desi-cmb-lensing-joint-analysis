import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------------------------------
# 0) Input files ----------------------------------------------------
lrg_files = [f"lrg_s0{i}_del.hpx2048.public.fits.gz" for i in range(1,5)]
mask_files = [f"masks/lrg_s0{i}_msk.hpx2048.public.fits.gz" for i in range(1,5)]

# ------------------------------------------------------------------
# 1) Read maps ------------------------------------------------------
lrg_maps = [hp.read_map(f) for f in lrg_files]
lrg_masks = [hp.read_map(f) for f in mask_files]

# ------------------------------------------------------------------
# 2) Apply masks ----------------------------------------------------
for m, mask in zip(lrg_maps, lrg_masks):
    bad = mask == 0
    m[bad] = hp.UNSEEN

# ------------------------------------------------------------------
# 3) Colormap -------------------------------------------------------
cmap = plt.get_cmap('coolwarm')
cmap.set_bad('gray')
cmap.set_under('white')

vmin, vmax = -0.5, 0.5

# ------------------------------------------------------------------
# 4) Figure ---------------------------------------------------------
fig = plt.figure(figsize=(14, 12))

for i, m in enumerate(lrg_maps):
    hp.mollview(
        m,
        sub=(2, 2, i+1),
        title=f"LRG bin z{i+1}",
        coord='G',
        cmap=cmap,
        min=vmin, max=vmax,
        notext=True,
    )

# 5) colorbar -------------------------------------------------------
cax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r'$\delta_{\mathrm{LRG}}$', fontsize=14, labelpad=8)

# 6) Save -----------------------------------------------------------
plt.savefig('desi_lrg_bins.png', dpi=300, bbox_inches='tight')
