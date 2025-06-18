import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d

# ------------------------------------------------------------------
# 0) Input file -----------------------------------------------------
# Path to the catalog containing RA [deg], DEC [deg] and Z columns
# The default value points to the catalog used to build the density maps.
cat_file = "dr9_lrg_pzbins_20230509.fits"

# Column names can vary depending on the catalog format. Adapt if needed.
ra_col = "RA"
dec_col = "DEC"
z_col = "Z"

# ------------------------------------------------------------------
# 1) Read catalog ---------------------------------------------------
cat = Table.read(cat_file)
ra = np.asarray(cat[ra_col])
dec = np.asarray(cat[dec_col])
z = np.asarray(cat[z_col])

# ------------------------------------------------------------------
# 2) Convert to 3D coordinates -------------------------------------
# Use the Planck15 cosmology to convert redshift to comoving distance in Mpc
r = cosmo.comoving_distance(z).value
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)

x = r * np.cos(dec_rad) * np.cos(ra_rad)
y = r * np.cos(dec_rad) * np.sin(ra_rad)
z_cart = r * np.sin(dec_rad)

# ------------------------------------------------------------------
# 3) Build redshift bins -------------------------------------------
bins = [0.4, 0.6, 0.8, 1.0, 1.2]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))

# ------------------------------------------------------------------
# 4) Figure --------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for i in range(4):
    msk = (z >= bins[i]) & (z < bins[i + 1])
    ax.scatter(
        x[msk], y[msk], z_cart[msk], s=1, color=colors[i], label=f"{bins[i]}<z<{bins[i+1]}"
    )

ax.set_xlabel("x [Mpc]")
ax.set_ylabel("y [Mpc]")
ax.set_zlabel("z [Mpc]")
ax.legend(loc="upper right", markerscale=5, fontsize=10)
ax.view_init(elev=25, azim=-60)

# ------------------------------------------------------------------
# 5) Save ----------------------------------------------------------
plt.tight_layout()
plt.savefig("desi_lrg_scatter_3d.png", dpi=300)
