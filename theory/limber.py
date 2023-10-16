import numpy as np
from scipy.integrate   import simps
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
    
class limb():
   """
   A class for computing Cgg and Ckg.

   MORE DOCS TO COME
   
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
   
   def __init__(self, dNdz, thy_fid, Pgm, Pgg, Pmm, background, lmax=1000, Nlval=64, zmin=0.001, zmax=2., Nz=50):
      """
      Parameters
      ----------
      dNdz_fname: str OR ndarray
         redshift distribution filename. 
      zmin: float
         ...
      zmax: float
         ...
      Nz: int
         ...
      """
      if isinstance(dNdz,str): dNdz = np.loadtxt(dNdz)
      self.Ng    = dNdz.shape[1] - 1
      self.zmin  = zmin
      self.zmax  = zmax
      self.Nz    = Nz
      self.z     = np.linspace(zmin,zmax,Nz)
        
      # evaluate dNdz on regular grid and normalize it so 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = interp1d(dNdz[:,0],dNdz[:,1:],axis=0,bounds_error=False,fill_value=0.)(self.z)
      self.dNdz  = self.dNdz.reshape((self.Nz,self.Ng))   # does nothing for Ng > 1

      self.l     = np.arange(lmax+1) 
      self.lval  = np.logspace(0,np.log10(lmax),Nlval)
      self.Nl    = len(self.l)
      self.Nlval = len(self.lval)
      # evaluate dNdz on regular grid and normalize it such 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = np.zeros((self.Nz,self.Ng))
      for j in range(self.Ng): self.dNdz[:,j] = Spline(dNdz[:,0],dNdz[:,j+1],ext=1)(self.z)
      norm       = simps(self.dNdz, x=self.z, axis=0)     # (Ng) ndarray
      norm       = self.gridMe(norm)
      self.dNdz /= norm
      # store theory predictions 
      self.Pgm         = Pgm
      self.Pgg         = Pgg
      self.Pmm         = Pmm
      self.background  = background
      # store fiducial cosmology (and set "current cosmology" to fiducial)
      self._thy_fid  = thy_fid
      # compute effective redshifts      
      self.computeZeff()

                          
   # Recompute effective redshifts whenever the 
   # fiducial cosmology changes 
   @property
   def thy_fid(self): return self._thy_fid
   @thy_fid.setter
   def thy_fid(self, new_thy_fid):
      self._thy_fid = new_thy_fid
      self.computeZeff()

   def computeZeff(self):
      """
      Computes the effective redshift for each galaxy sample 
      assuming the fiducial cosmology and saves them to 
      self.zeff, which is a (Ng) ndarray. If no background 
      theory has been supplied, sets self.zeff = None.
      """
      OmM,chistar,Ez,chi = self.background(self.thy_fid,self.z)
      _,Wg,_             = self.projectionKernels(self.thy_fid)
      def zeff(i):
         denom  = np.trapz(Wg[:,i]*Wg[:,i]/chi**2,x=chi)
         numer  = np.trapz(Wg[:,i]*Wg[:,i]*self.z/chi**2,x=chi)
         return numer/denom
      self.zeff = np.array([zeff(i) for i in range(self.Ng)])

    
   def evaluate(self, i, thy_args, verbose=True):
      """
      Computes background quantities, projection kernels,
      and power spectra for a given cosmology. Returns
      
      chi          # comoving distance, (Nz) ndarray
      Wk           # CMB lensing kernel, (Nz) ndarray
      Wg_clust     # galaxy clustering kernels, (Nz,Ng) ndarray
      Wg_mag       # galaxy magnification kernels, (Nz,Ng) ndarray
      Pgm_eval     # Pgm tables at each effective z, (Ng,Nk,1+Nmono) ndarray
      Pgg_eval     # Pgm tables at each effective z, (Ng,Nk,1+Nmono) ndarray
      Pmm_eval     # Pmm evaluated at each z in self.z, (Nk,1+Nz) ndarray

      Nmono is the number of monomials (e.g. 1, alpha0, ...), which can in 
      general be different for Pgm and Pgg. The "+1" is a column of ks.
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
      verbose: bool, default=True
         when True, prints message when theory code 
         (Pgg, Pgm, Pmm, or background) is missing
      """
      OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      Wk,Wg_clust,Wg_mag = self.projectionKernels(thy_args,bkgrnd=[OmM,chistar,Ez,chi])
      Pgm_eval = self.Pgm(thy_args,self.zeff[i])
      Pgg_eval = self.Pgg(thy_args,self.zeff[i])
      Pmm_eval = self.Pmm(thy_args,self.z)
      return chi,Wk,Wg_clust,Wg_mag,Pgm_eval,Pgg_eval,Pmm_eval

       
   def gridMe(self,x):
      """
      Places input on a (Nz,Ng) grid. If x is z-independent, 
      repeats Ng times. If x is galaxy-independent, repeats 
      Nz times. If x is a float, repeats Nz*Ng times. 
      
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
      N = x.shape[0]
      if N == self.Ng:
         return np.tile(x,self.Nz).reshape((self.Nz,self.Ng))
      if N == self.Nz:
         return np.repeat(x,self.Ng).reshape((self.Nz,self.Ng))
      else: 
         s = 'input must satisfy len = self.Ng or self.Nz'
         raise RuntimeError(s)
      
      
   def projectionKernels(self, thy_args, bkgrnd=None):
      """
      Computes the projection kernels [h/Mpc] for CMB lensing 
      and each galaxy sample. The CMB lensing kernel (Wk) is 
      a (Nz) ndarray. The galaxy kernels are (Nz,Ng) ndarrays. 
      The full galaxy kernel is 
               Wg = Wg_clust + (5*s-2) * Wg_mag
      where s is the slop of the cumulative magnitude func. 
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
         
      Raises
      ------
      RuntimeError
         if self.background is None
      """
      if self.background is None:
         s  = 'must provide a background code to compute projection kernels'
         raise RuntimeError(s)
         
      OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      H0                 = 100./299792.458 # [h/Mpc] units
      if bkgrnd is None: OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      else:              OmM,chistar,Ez,chi = bkgrnd
      H0 = 100./299792.458 # [h/Mpc] units
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
         # approximates \int_z^{zstar} dz' x(z') with a Riemann sum
         x = np.flip(x,axis=0)
         x = np.cumsum(x,axis=0) * (self.z[1]-self.z[0])
         return np.flip(x,axis=0)
      Wg_mag  = self.gridMe(chi)*integrate_z_zstar(self.dNdz)
      Wg_mag -= self.gridMe(chi**2)*integrate_z_zstar(self.gridMe(1./chi)*self.dNdz)
      Wg_mag *= self.gridMe(1.5*OmM*H0**2.*(1.+self.z))
      
      return Wk,Wg_clust,Wg_mag


   def computeCggCkg(self, i, thy_args, smag, ext=3):
      """
      """
      # Evaluate projection kernels and power spectra.
      # The "kgrid" is defined such that kgrid[i,j] = (l[j]+0.5)/chi(z[i])
      chi,Wk,Wg_clust,Wg_mag,PgmT,PggT,PmmT = self.evaluate(i, thy_args)                 
      kgrid = (np.tile(self.lval+0.5,self.Nz)/np.repeat(chi,self.Nlval)).reshape((self.Nz,self.Nlval))
      
      Wg_clust   = Wg_clust[:,i]    # (Nz) ndarray
      Wg_mag     = Wg_mag[:,i]      # (Nz) ndarray
      Nmono_auto = PggT.shape[1]-1  # number of monomials for auto
      Nmono_cros = PgmT.shape[1]-1  # number of monomials for cross
      
      # interpolate
      PggIntrp = np.zeros(kgrid.shape+(Nmono_auto,))
      PgmIntrp = np.zeros(kgrid.shape+(Nmono_cros,))
      for j in range(Nmono_auto): PggIntrp[:,:,j] = Spline(PggT[:,0],PggT[:,j+1],ext=ext)(kgrid)
      for j in range(Nmono_cros): PgmIntrp[:,:,j] = Spline(PgmT[:,0],PgmT[:,j+1],ext=ext)(kgrid)
      Pgrid = np.zeros((self.Nz,self.Nlval)) # kgrid.shape
      for j in range(self.Nz): 
         Pgrid[j,:] = Spline(PmmT[:,0],PmmT[:,j+1],ext=1)(kgrid[j,:])
          
      # assume that mono_auto = 1, auto1, auto2, ... AND ADD SHOT NOISE
      # and that    mono_cros = 1, cros1, cros2, ...
      Nmono_tot = 1 + Nmono_auto + (Nmono_cros-1)
      def reshape_kernel(kernel): return np.repeat(kernel/chi**2.,self.Nlval).reshape(kgrid.shape)    
          
      ##### Cgg
      Cgg = np.ones((self.Nl,Nmono_tot))
      # the "1" piece
      integrand  = reshape_kernel(Wg_clust**2)                  * PggIntrp[:,:,0]
      integrand += 2*(5*smag-2)*reshape_kernel(Wg_mag*Wg_clust) * PgmIntrp[:,:,0]
      integrand += (5*smag-2)**2*reshape_kernel(Wg_mag**2)      * Pgrid
      integral   = simps(integrand,x=chi,axis=0)
      Cgg[:,0]   = Spline(self.lval,integral)(self.l)
      # the mono_auto pieces
      for j in range(Nmono_auto-1):
         integrand  = reshape_kernel(Wg_clust**2) * PggIntrp[:,:,j+1]
         integral   = simps(integrand,x=chi,axis=0)
         Cgg[:,j+1] = Spline(self.lval,integral)(self.l)
      # adding shot noise (already ones)
      # the mono_cros pieces
      for j in range(Nmono_cros-1):
         integrand = 2*(5*smag-2)*reshape_kernel(Wg_clust*Wg_mag) * PgmIntrp[:,:,j+1]
         integral  = simps(integrand,x=chi,axis=0)
         Cgg[:,j+1+Nmono_auto] = Spline(self.lval,integral)(self.l)
      
      ##### Ckg
      Ckg = np.zeros((self.Nl,Nmono_tot))
      # the "1" piece
      integrand  = reshape_kernel(Wk*Wg_clust)          * PgmIntrp[:,:,0]
      integrand += (5*smag-2)*reshape_kernel(Wk*Wg_mag) * Pgrid
      integral   = simps(integrand,x=chi,axis=0)
      Ckg[:,0]   = Spline(self.lval,integral)(self.l)  
      # the mono_auto pieces are zero (including shot noise)          
      # the mono_cros pieces
      for j in range(Nmono_cros-1):
         integrand = reshape_kernel(Wk*Wg_clust) * PgmIntrp[:,:,j+1]
         integral  = simps(integrand,x=chi,axis=0) 
         Ckg[:,j+1+Nmono_auto] = Spline(self.lval,integral)(self.l)
          
      return Cgg,Ckg


   def computeCgigjZevolution(self, i, j, thy_args, mono_auto, mono_cross, smag, ext=3):
      """
      i: i'th sample
      j: j'th sample
      thy_args, mono_auto and mono_corss, and smag are all functions of z
      
      does not add shot noise by default
      """
      # Evaluate projection kernels and power spectra.
      # The "kgrid" is defined such that kgrid[i,j] = (l[j]+0.5)/chi(z[i])
      thy_args_ = thy_args(self.zeff[i])
      OmM,chistar,Ez,chi = self.background(thy_args_,self.z)
      Wk,Wg_clust,Wg_mag = self.projectionKernels(thy_args_,bkgrnd=[OmM,chistar,Ez,chi])
      PmmT  = self.Pmm(thy_args_,self.z)              
      kgrid = (np.tile(self.lval+0.5,self.Nz)/np.repeat(chi,self.Nlval)).reshape((self.Nz,self.Nlval))
      
      PgmT = np.zeros_like(PmmT); PgmT[:,0] = PmmT[:,0].copy()
      PggT = np.zeros_like(PmmT); PggT[:,0] = PmmT[:,0].copy()
      for k,z in enumerate(self.z):
         monx        = np.array([1.]+list(mono_cross(z)))
         mona        = np.array([1.]+list(mono_auto(z)))
         PgmT[:,k+1] = np.dot(self.Pgm(thy_args(z),z)[:,1:],monx)
         PggT[:,k+1] = np.dot(self.Pgg(thy_args(z),z)[:,1:],mona)   
          
      Wgi_clust   = Wg_clust[:,i]    # (Nz) ndarray
      Wgi_mag     = Wg_mag[:,i]      # (Nz) ndarray
      Wgj_clust   = Wg_clust[:,j]    # (Nz) ndarray
      Wgj_mag     = Wg_mag[:,j]      # (Nz) ndarray
              
      PgmGrid = np.zeros_like(kgrid)   
      PggGrid = np.zeros_like(kgrid) 
      PmmGrid = np.zeros_like(kgrid) 
      for k in range(self.Nz): 
         PgmGrid[k,:] = Spline(PgmT[:,0],PgmT[:,k+1],ext=1)(kgrid[k,:])   
         PggGrid[k,:] = Spline(PggT[:,0],PggT[:,k+1],ext=1)(kgrid[k,:])   
         PmmGrid[k,:] = Spline(PmmT[:,0],PmmT[:,k+1],ext=1)(kgrid[k,:])
          
      def reshape_kernel(kernel): return np.repeat(kernel/chi**2.,self.Nlval).reshape(kgrid.shape)    
          
      ##### Cgigj
      integrand  = reshape_kernel(Wgi_clust*Wgj_clust)                   * PggGrid
      integrand += reshape_kernel((5*smag(zelf.z)-2)*Wgi_mag*Wgj_clust)  * PgmGrid
      integrand += reshape_kernel((5*smag(zelf.z)-2)*Wgj_mag*Wgi_clust)  * PgmGrid
      integrand += reshape_kernel((5*smag(zelf.z)-2)**2*Wgi_mag*Wgj_mag) * PmmGrid
      integral   = simps(integrand,x=chi,axis=0)
      Cgigj      = Spline(self.lval,integral)(self.l)
          
      return Cgigj