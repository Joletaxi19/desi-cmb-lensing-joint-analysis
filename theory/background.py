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
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi
   
import jax.numpy as jnp
   
def analyticBackground(thy_args, zs):
   """
   TO DO: MAKE THIS MORE ACCURATE, BUT FOR NOW 
   THIS IS FINE FOR TESTING CODE
   """
   omb,omc,ns,As,H0 = thy_args
   OmM = (omb+omc)/(H0/100.)**2.
   def E_z(z): return jnp.sqrt(OmM*(1.+z)**3. + (1.-OmM))
   Ez = E_z(zs)
   # chistar obviously depends on cosmology, 
   # but for now let's ignore that for the sake of testing
   chistar = 9400. 
   #def chi_z(z):
   #   zint = jnp.linspace(0.,z,50)
   #   return jnp.trapz(2997.92458/E_z(zint),zint)
   #chi = jnp.array([chi_z(z) for z in zs])

   # not even close, just want it to be fast for now
   chi = 1000.*zs
   return OmM,chistar,Ez,chi