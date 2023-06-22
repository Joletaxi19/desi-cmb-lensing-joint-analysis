# This is a wrapper around various codes for computing background 
# quantities (OmM, chistar, Ez, chi) relevant for Limber integrals.
# Each method should have thy_args and zs as its first two arguments.
#
# Currently wrapped:
# - background from CLASS

# ingredients
import numpy as np
from classy import Class

def classyBackground(self, thy_args, zs):
   """
   Computes background quantities relevant for Limber integrals
   using CLASS. Returns OmM (~0.3), chistar (comoving dist [h/Mpc] 
   to the surface of last scatter), Ez (H(z)/H0 evaluated on self.z), 
   and chi (comoving distance [h/Mpc] evaluated on self.z).
   
   Parameters
   ----------
   thy_args: dict
      inputs to CLASS for computing background quantities
   zs: list OR ndarray
      redshifts to evaluate chi(z) and E(z) 
   """
   cosmo = Class()
   cosmo.set(thy_args)
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi
