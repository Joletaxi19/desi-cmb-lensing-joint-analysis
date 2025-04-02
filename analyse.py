import healpy as hp
import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt


# Chemin d'accès aux fichiers extraits
alm_file = "maps/baseline/kappa_alm_data_act_dr6_lensing_v1_baseline.fits"
mask_file = "maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"

# Lecture des multipôles
alm = hp.read_alm(alm_file)

# Filtrage : pour l > 3000, mettre à zéro
lmax = hp.Alm.getlmax(len(alm))

ell = np.arange(lmax+1)
alm_filtered = np.copy(alm).astype(np.complex128)
for ell_val in range(3001, lmax+1):
    m_start = hp.Alm.getidx(lmax, ell_val, 0)
    m_end = hp.Alm.getidx(lmax, ell_val, ell_val) + 1
    alm_filtered[m_start:m_end] = 0.0j

# on downgrade à nside 2048 en utilisant ud_grade
nside_target = 2048 

kappa_map_full = hp.alm2map(alm_filtered, nside=4096)
mask_full = hp.read_map(mask_file)

kappa_map = hp.ud_grade(kappa_map_full, nside_out=nside_target)
mask = hp.ud_grade(mask_full, nside_out=nside_target)

# Appliquer le masque
kappa_map_masked = hp.ma(kappa_map, mask)
kappa_map_masked.mask = np.logical_not(mask.astype(bool))

# Calculer la puissance spectrale

field_kappa = nmt.NmtField(mask, [kappa_map_masked.filled()], lmax=3000)

# binning
nside = nside_target
ell_edges = np.linspace(20, 3001, 70)
ell_edges = np.round(ell_edges).astype(int)
ell_edges = np.unique(ell_edges)
binning = nmt.NmtBin.from_edges(ell_edges[:-1], ell_edges[1:])

cl_coupled = nmt.compute_coupled_cell(field_kappa, field_kappa)

workspace = nmt.NmtWorkspace()
workspace.compute_coupling_matrix(field_kappa, field_kappa, binning)
cl_decoupled = workspace.decouple_cell(cl_coupled)

ells_eff = binning.get_effective_ells()

# Affichage avec échelle linéaire en x et logarithmique en y
plt.figure(figsize=(8, 5))
plt.semilogy(ells_eff, np.abs(cl_decoupled[0]), 'o-', label=r'$C_\ell^{\kappa\kappa}$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell$')
plt.title("Spectre de puissance de la convergence CMB (ACT DR6)")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.show()