import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from classy import Class

#################################
##
## Some general purpose functions
##
#################################

def effectiveRedshift(chi, WA, WB, zOfchi):
   """
   Computes the effective redshift.
   
   Parameters
   ----------
   chi: (N) ndarray
      comoving distance [Mpc/h]. chi[0] and chi[-1]
      correspond to the integration limits. 
   WA: (N) ndarray
      projection function [h/Mpc] for tracer A
   WB: (N) ndarray
      projection function [h/Mpc] for tracer B
   zOfchi: (N) ndarray
      redshift as a function of comoving distance
   """
   integrand  = WA*WB/chi**2
   denom      = np.trapz(integrand,x=chi)
   integrand  = WA*WB*zOfchi/chi**2
   numer      = np.trapz(integrand,x=chi)
   return numer/denom
   

def limberWithoutEvolution(chi, WA, WB, pAB):
   """
   Computes the Limber integral assuming a redshift-
   independent power spectrum.
   
   Parameters
   ----------
   chi: (N) ndarray
      comoving distance [Mpc/h]. chi[0] and chi[-1]
      correspond to the integration limits. 
   WA: (N) ndarray
      projection function [h/Mpc] for tracer A
   WB: (N) ndarray
      projection function [h/Mpc] for tracer B
   pAB: (N,L) ndarray
      pAB[:,i] corresponds to the power spectrum
      evaluated at k = (l[i]+0.5)/chi
   """
   N,L = pAB.shape
   # create L copies of WA*WB/chi^2
   integrand  = WA*WB/chi**2.
   integrand  = np.repeat(integrand,L).reshape((N,L))
   integrand *= pAB
   return np.trapz(integrand,x=chi,axis=0)
   
   
def limberWithEvolution(chi, WA, WB, pAB, chiEdges):
   """
   Computes the Limber integral, taking into account
   the redshift-dependence of the power spectrum.
   
   Parameters
   ----------
   chi: (N) ndarray
      comoving distance [Mpc/h]
   WA: (N) ndarray
      projection function [h/Mpc] for tracer A
   WB: (N) ndarray
      projection function [h/Mpc] for tracer B
   pAB: (N,L,Z-1) ndarray
      pAB[:,i,k] corresponds to the power spectrum
      at redshift Z[k] evaluated at k = (l[i]+0.5)/chi
   chiEdges: (Z) ndarray
      edges of each 'redshift bin' in comoving
      distance units
      
   Raises
   ------
   RuntimeError
      If chiEdges extends beyond the range of chi
   """
   if (chiEdges[0]<chi[0]) or (chiEdges[-1]>chi[-1]):
      s= 'chiEdges extends beyond the domain of chi'
      raise RuntimeError(s)
   
   N,L,Z = pAB.shape
   Z    += 1
   res   = np.zeros(L)
   for z in Z:
      edgeLo = chiEdges[z]
      edgeHi = chiEdges[z+1]
      I = np.where((chi >= edgeLo) & (chi < edgeHi+1e-3))
      res += limberWithoutEvolution(chi[I], WA[I], WB[I], pAB[I,:,z])
   return res


####################################
##
## A class for computing Cgg and Ckg
## neatly
##
####################################

class galKapCl():
   """
   A class for computing Cgg and Ckg.

   BETTER DESCRIPTION TO COME
   
   ...
   
   Attributes
   ----------
   z: (Nz) ndarray
      ...
   dNdz: (Nz,Ng) ndarray
      redshift distribution of the galaxy sample(s)
   Ng: int
      number of galaxy samples
      
   Methods
   -------
   XXXXX
   """
   
   def __init__(self, dNdz_fname, zmin=0.001, zmax=5., Nz=500,
                      Pgm=None, Pgg=None, Pmm=None, thy_fid={}):
      """
      Parameters
      ----------
      dNdz_fname: str
         redshift distribution filename. 
      zmin: float
         ...
      zmax: float
         ...
      Nz: int
         ...
      """
      dNdz       = np.loadtxt(dNdz_fname)
      self.Ng    = dNdz.shape[1] - 1
      self.zmin  = zmin
      self.zmax  = zmax
      self.Nz    = Nz
      self.z     = np.linspace(zmin,zmax,Nz)
      # evaluate dNdz on regular grid and normalize it so 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = interp1d(dNdz[:,0],dNdz[:,1:],axis=0,bounds_error=False,fill_value=0.)(self.z)
      self.dNdz  = self.dNdz.reshape((self.Nz,self.Ng))   # does nothing for Ng > 1
      norm       = simps(self.dNdz, x=self.z, axis=0)     # (Ng) ndarray
      norm       = self.gridMe(norm)
      self.dNdz /= norm
      # compute effective redshifts (for Pgg about the fiducial cosmology)  
      self.thy_fid = thy_fid
      OmM,chistar,Ez,chi = self.background(thy_fid)
      Wk,Wg_clust,Wg_mag = self.projectionKernels(thy_fid)
      zeff = lambda i: effectiveRedshift(chi, Wg_clust[:,i], Wg_clust[:,i], self.z)
      self.zeff = np.array([zeff(i) for i in range(self.Ng)])
      # store theory prediction 
      self.Pgm  = Pgm
      self.Pgg  = Pgg
      self.Pmm  = Pmm


   def gridMe(self,x):
      """
      Places input on a (Nz,Ng) grid. If x is z-independent, 
      repeats Ng times. If x is galaxy-independent, repeats 
      Nz times. If x is a float, repeats (Nz,Ng) times. 
      
      Parameters
      ----------
      x: float, (Nz) ndarray, OR (Ng) ndarray
         the input to be gridded
         
      Raises
      ------
      RuntimeError
         if not [(x is float) or (x is 1-D 
         ndarray with len = Nz or Ng)] 
      """
      if isinstance(x,float):
         return x*np.ones_like(self.dNdz)
      if not isinstance(x,np.ndarray):
         s = 'input must either be a float or 1D ndarray'
         raise RuntimeError(s)
      if len(list(x.shape))>1:
         s = 'input should only have 1 dimension'
         raise RuntimeError(s)
      N = x.shape[0]
      if N == self.Ng:
         return np.tile(x,self.Nz).reshape((self.Nz,self.Ng))
      if N == self.Nz:
         return np.repeat(x,self.Ng).reshape((self.Nz,self.Ng))
      else: 
         s = 'input must satisfy len = self.Ng or self.Nz'
         raise RuntimeError(s)
      
      
   def background(self, thy_args):
      """
      Computes background quantities relevant for Limber integrals. 
      Returns OmM (~0.3), chistar (comoving dist [h/Mpc] to the 
      surface of last scatter), Ez (H(z)/H0 evaluated on self.z), 
      and chi (comoving distance [h/Mpc] evaluated on self.z).
      
      Parameters
      ----------
      thy_args: dict
         inputs to CLASS for computing background quantities
      """
      cosmo = Class()
      cosmo.set(thy_args)
      cosmo.compute()
      
      OmM     = cosmo.Omega0_m()
      zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
      chistar = cosmo.comoving_distance(zstar)*cosmo.h()
      Ez      = np.vectorize(cosmo.Hubble)(self.z)/cosmo.Hubble(0.)
      chi     = np.vectorize(cosmo.comoving_distance)(self.z)*cosmo.h()
      
      return OmM,chistar,Ez,chi
      
      
   def projectionKernels(self, thy_args):
      """
      Computes the projection kernels [h/Mpc] for CMB lensing 
      and each galaxy sample. The CMB lensing kernel (Wk) is 
      a (Nz) ndarray. The galaxy kernels are (Nz,Ng) ndarrays. 
      The full galaxy kernel is 
               Wg = Wg_clust + (5*s-2) * Wg_mag
      where s is the slop of the cumulative magnitude func. 
      
      Parameters
      ----------
      thy_args: dict
         inputs to CLASS for computing background quantities
      """
      OmM,chistar,Ez,chi = self.background(thy_args)
      H0                 = 100./299792.458 # [h/Mpc] units
      ## CMB lensing
      Wk  = 1.5*OmM*H0**2.*(1.+self.z)
      Wk *= chi*(chistar-chi)/chistar
      ## Galaxies
      # clustering contribution
      Wg_clust  = self.gridMe(H0*Ez) * self.dNdz  
      # magnification bias contribution
      def integrate_z_zstar(x):
         # approximates the integral 
         # \int_z^{zstar} dz' x(z') 
         # with a Riemann sum
         x = np.flip(x,axis=0)
         x = np.cumsum(x,axis=0) * (self.z[1]-self.z[0])
         return np.flip(x,axis=0)
      Wg_mag  = self.gridMe(chi)*integrate_z_zstar(self.dNdz)
      Wg_mag -= self.gridMe(chi**2)*integrate_z_zstar(self.gridMe(1./chi)*self.dNdz)
      Wg_mag *= self.gridMe(1.5*OmM*H0**2.*(1.+self.z))
      
      return Wk,Wg_clust,Wg_mag
  
   """
   TODO: decide how I want to organize the Pk -> Cl calculation
   
   magnification bias should probably not be included as column in tables (it
   multiplies b1,etc. and Cgg is quadratic in smag so no analytic marg). However
   changing smag should always be fast, don't need to redo full Pk -> Cl calculation.
   """
