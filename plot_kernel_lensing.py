import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

# Redshift grid
z = np.linspace(0.01, 5, 500)

# Comoving distances in Mpc
chi = cosmo.comoving_distance(z).to(u.Mpc).value  
chi_star = cosmo.comoving_distance(1100).to(u.Mpc).value  

# Hubble parameter H(z) in km/s/Mpc
H_z = cosmo.H(z).value  

# Compute the lensing efficiency kernel (up to a constant factor)
W_kappa = (1+z)/H_z * chi * (1 - chi/chi_star)
W_kappa /= np.max(W_kappa)  # normalization for visualization

# Plotting
plt.figure(figsize=(8,5))
plt.plot(z, W_kappa, label=r'$W^\kappa(z)$ (CMB lensing kernel)')
plt.xlabel('Redshift z')
plt.ylabel('Normalized lensing efficiency')
plt.xlim(0, 5)
plt.title('CMB Lensing Efficiency Kernel (with H(z))')
plt.legend()
plt.grid(True)
plt.savefig('lensing_kernel.png', dpi=300)
plt.show()