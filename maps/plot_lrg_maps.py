import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize

# ------------------------------------------------------------------
# 0) Input files ----------------------------------------------------
lrg_files = [f"lrg_s0{i}_del.hpx2048.public.fits.gz" for i in range(1,5)]
mask_files = [f"masks/lrg_s0{i}_msk.hpx2048.public.fits.gz" for i in range(1,5)]

# ------------------------------------------------------------------
# 1) Read maps ------------------------------------------------------
lrg_maps = []
lrg_masks = []

for lrg_file, mask_file in zip(lrg_files, mask_files):
    lrg_map = hp.read_map(lrg_file)
    mask = hp.read_map(mask_file)
    
    # Apply mask directly
    bad = mask == 0
    lrg_map[bad] = hp.UNSEEN
    
    lrg_maps.append(lrg_map)
    lrg_masks.append(mask)

# ------------------------------------------------------------------
# 2) Calculate color scale with enhanced contrast -------------------
valid_data = []
for m, mask in zip(lrg_maps, lrg_masks):
    valid = mask != 0
    valid_data.extend(m[valid])

valid_data = np.array(valid_data)

vmax = np.percentile(np.abs(valid_data), 95)
vmin = -vmax

# ------------------------------------------------------------------
# 3) Create custom colormap for better contrast ---------------------
colors_r = plt.cm.Blues_r(np.linspace(0.2, 0.8, 128))
colors_b = plt.cm.Reds(np.linspace(0.2, 0.8, 128))
colors = np.vstack((colors_r, colors_b))
custom_cmap = ListedColormap(colors)
custom_cmap.set_bad('lightgray', alpha=0.5)
custom_cmap.set_under('white')
custom_cmap.set_over('purple')

# ------------------------------------------------------------------
# 4) Create figure -------------------------------------------------
fig = plt.figure(figsize=(18, 12))

titles = [
    "LRG bin 1 (0.4 < z < 0.6)",
    "LRG bin 2 (0.6 < z < 0.8)", 
    "LRG bin 3 (0.8 < z < 1.0)",
    "LRG bin 4 (1.0 < z < 1.2)"
]
positions = [221, 222, 223, 224]

# ------------------------------------------------------------------
# 5) Plot each map with enhanced contrast --------------------------
for i, (m, pos, title) in enumerate(zip(lrg_maps, positions, titles)):
    hp.mollview(
        map=m,
        fig=fig.number,
        sub=pos,
        title=title,
        cmap=custom_cmap,
        min=vmin,
        max=vmax,
        cbar=False,
        notext=True,
        coord="G",
        rot=(0, 0, 0)
    )
    
    hp.graticule(dpar=20, dmer=30, color='gray', alpha=0.3)

# ------------------------------------------------------------------
# 6) Add colorbar with ticks ---------------------------------------
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])

norm = Normalize(vmin=vmin, vmax=vmax)
cb = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=custom_cmap),
    cax=cbar_ax,
    orientation='horizontal',
    extend='both'
)
# Ajouter plus de graduations pour mieux interpréter les valeurs
tick_positions = np.linspace(vmin, vmax, 9)
cb.set_ticks(tick_positions)
cb.set_ticklabels([f"{x:.2f}" for x in tick_positions])
cb.set_label(r'$\delta_{\mathrm{LRG}}$ (surdensité)', fontsize=18)
cb.ax.tick_params(labelsize=12)

# Améliorer l'espacement pour une meilleure lisibilité
plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.92, bottom=0.1)

# Sauvegarder en haute résolution
plt.savefig('desi_lrg_bins.png', dpi=300, bbox_inches='tight')

print("Figures sauvegardées avec contraste amélioré!")
print("Version principale: 'desi_lrg_bins.png'")