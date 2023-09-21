# This is a wrapper around various codes for computing background 
# quantities (OmM, chistar, Ez, chi) relevant for Limber integrals.
# Each method should have thy_args and zs as its first two arguments.
#
# Currently wrapped:
# - background from CLASS

# ingredients
import numpy as np
from classy import Class

def classyBackground(thy_args, zs):
   """
   Computes background quantities relevant for Limber integrals
   using CLASS. Returns OmM (~0.3), chistar (comoving dist [h/Mpc] 
   to the surface of last scatter), Ez (H(z)/H0 evaluated on zs), 
   and chi (comoving distance [h/Mpc] evaluated on zs).
   
   Parameters
   ----------
   thy_args: dict
      inputs to CLASS for computing background quantities
   zs: list OR ndarray
      redshifts to evaluate chi(z) and E(z) 
   """
   cosmo = Class()
   cosmo.set(thy_args)
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
             
   params = {'A_s': 1e-10*np.exp(ln10As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   
   cosmo = Class()
   cosmo.set(params)
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi