import numpy as np
from scipy.integrate   import simps
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

#################################
##
## Some general purpose functions
##
#################################

def effRedshift(chi, WA, WB, zOfchi):
   """
   Computes the effective redshift. Returns float.
   
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
   
def limberIntegral(chi, WA, WB, pAB):
   """
   Computes the Limber integral. Returns (L) ndarray.
   
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
      pAB[i,j] corresponds to the AB power spectrum
      evaluated at k = (l[j]+0.5)/chi(z[i])
   """
   N,L = pAB.shape
   # create L copies of WA*WB/chi^2
   integrand  = WA*WB/chi**2.
   integrand  = np.repeat(integrand,L).reshape((N,L))
   integrand *= pAB
   return np.trapz(integrand,x=chi,axis=0)
   
####################################
##
## A class for computing Cgg and Ckg
## neatly
##
####################################

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
   
   def __init__(self, dNdz_fname, zmin=0.001, zmax=5., Nz=500,
                      Pgm=None, Pgg=None, Pmm=None, 
                      background=None, thy_fid={}):
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
      # store theory predictions 
      self.Pgm         = Pgm
      self.Pgg         = Pgg
      self.Pmm         = Pmm
      self._background = background
      # store fiducial cosmology (and set "current cosmology" to fiducial)
      self._thy_fid  = thy_fid
      self._thy_args = thy_fid.copy()
      # compute effective redshifts and set up fidicual kernels & power spectra
      # if applicable (when the relevant theory codes are supplied)

      # RIGHT NOW evaluate WILL JUST THROW AN ERROR IF PK CODES ARE NOT GIVEN, MAKE THAT ERROR HANDELING MORE CLEVER
                          
      self.computeZeff()
      self.evaluate(thy_fid)

                          
   # Recompute effective redshifts whenever the 
   # fiducial cosmology changes 
   @property
   def thy_fid(self): return self._thy_fid
   @thy_fid.setter
   def thy_fid(self, new_thy_fid):
      self._thy_fid = new_thy_fid
      self.computeZeff()
   # or when the background theory code changes
   @property     
   def background(self): return self._background
   @background.setter
   def background(self, newBackground):
      self._background = newBackground
      self.computeZeff()

   # When the cosmology changes, recompute
   # projection kernels and power spectra
   @property
   def thy_args(self): return self._thy_args        
   @thy_args.setter
   def thy_args(self, new_thy_args):
      if new_thy_args != self.thy_args:
         self.evaluate(new_thy_args)
         self._thy_args = new_thy_args

    
   def computeZeff(self):
      """
      Computes the effective redshift for each galaxy sample 
      assuming the fiducial cosmology and saves them to 
      self.zeff, which is a (Ng) ndarray. If no background 
      theory has been supplied, sets self.zeff = None.
      """
      if self._background is None: 
          print('No background code given, skipping zeff caclaultion.')
          self.zeff = None ; return ''
      OmM,chistar,Ez,chi = self.background(self.thy_fid,self.z)
      _,Wg,_             = self.projectionKernels(self.thy_fid)
      zeff = lambda i: effRedshift(chi, Wg[:,i], Wg[:,i], self.z)
      self.zeff = np.array([zeff(i) for i in range(self.Ng)])

    
   def evaluate(self, thy_args, verbose=True):
      """
      Computes background quantities, projection kernels,
      and power spectra for a given cosmology. These
      are stored as:
      
      self.chi          # comoving distance, (Nz) ndarray
      self.Wk           # CMB lensing kernel, (Nz) ndarray
      self.Wg_clust     # galaxy clustering kernels, (Nz,Ng) ndarray
      self.Wg_mag       # galaxy magnification kernels, (Nz,Ng) ndarray
      self.Pgm_eval     # Pgm tables at each effective z, (Ng,Nk,1+Nmono) ndarray
      self.Pgg_eval     # Pgm tables at each effective z, (Ng,Nk,1+Nmono) ndarray
      self.Pmm_eval     # Pmm evaluated at each z in self.z, (Nk,1+Nz) ndarray

      Nmono is the number of monomials (e.g. 1, b1, b2, ...), which can in 
      general be different for Pgm and Pgg. The "+1" is a column of ks.
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
      verbose: bool, default=True
         when True, prints message when theory code 
         (Pgg, Pgm, Pmm, or background) is missing
      """
      def pr(s): 
         if verbose: print(s)
      # background
      if self.background is None:
          pr('No background code given, skipping evaluate'); return ''
      _,_,_,self.chi                    = self.background(thy_args,self.z)
      self.Wk,self.Wg_clust,self.Wg_mag = self.projectionKernels(thy_args)
      # Pgm
      if self.Pgm is None: pr('No Pgm provided, skipping Pgm calculation')
      else: self.Pgm_eval = np.array([self.Pgm(thy_args,z) for z in self.zeff])
      # Pgg
      if self.Pgg is None: pr('No Pgg provided, skipping Pgg calculation')
      else: self.Pgg_eval = np.array([self.Pgg(thy_args,z) for z in self.zeff])
      # Pmm
      if self.Pmm is None: pr('No Pmm provided, skipping Pmm calculation')
      else: self.Pmm_eval = self.Pmm(thy_args,self.z)

       
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
  
  
   def computeCkg(self, i, thy_args, lmax=500):
      """
      Computes the CMB lensing-galaxy cross-correlation
      up to a specified lmax within the Limber approximation.
      Returns a clustering contribution Ckg_clust in the form 
      of a "monomial table", i.e. a (Nl,Nmono) ndarray, as well 
      as the magnification contribution Ckg_mag, which is a (Nl) 
      ndarray. Given a set of monomial coefficients "coeff", 
      which is a (Nmono) ndarray, the full prediction is: 
        Ckg = np.dot(Ckg_clust,coeff) + (5*s-2) * Ckg_mag
      where s is the slop of the cumulative magnitude func.  

      Parameters
      ----------
      i: int
         index of galaxy sample, i.e. i = 0,1,...,Ng-1
      thy_args: type can vary according to theory codes
         cosmological inputs
      lmax: int, default=500
         maximum multipole (inclusive)
      """
      # Compute power spectra (if necessary) by 
      # setting thy_args. The "kgrid" is defined
      # such that kgrid[i,j] = (l[j]+0.5)/chi[i]
      self.thy_args = thy_args
      l     = np.arange(lmax+1) ; Nl = len(l)
      kgrid = (np.tile(l+0.5,self.Nz)/np.repeat(self.chi,Nl)).reshape((self.Nz,Nl))
       
      # "clustering contribution"
      k         = self.Pgm_eval[i][:,0]     # assume that fist col. = k
      Pgm       = self.Pgm_eval[i][:,1:]    # (Nk,Nmono) ndarray
      Nmono     = Pgm.shape[1]              # number of monomials
      Ckg_clust = np.zeros((Nl,Nmono))
      for j in range(Nmono):
         pint = Spline(k,Pgm[:,j],ext=1)
         Ckg_clust[:,j] = limberIntegral(self.chi, self.Wk, self.Wg_clust[:,i], pint(kgrid))
      
      # magnification contribution
      k       = self.Pmm_eval[:,0]          # assume that fist col. = k
      Pmm     = self.Pmm_eval[:,1:]         # (Nk,Nz) ndarray
      Pgrid   = np.zeros((self.Nz,Nl))
      for j in range(self.Nz):
          Pgrid[j,:] = Spline(k,Pmm[:,j],ext=1)(kgrid[j,:])
      Ckg_mag = limberIntegral(self.chi, self.Wk, self.Wg_mag[:,i], Pgrid)

      return Ckg_clust, Ckg_mag

    
   def computeCgg(self, i, thy_args, lmax=500):
      """
      Computes the galaxy auto-correlation up to a specified 
      lmax within the Limber approximation. Returns a clustering 
      contribution Cgg_clust in the form of a "monomial table", i.e. 
      a (Nl,Nmono_auto) ndarray, the linear magnification contribution 
      Cgg_mag_lin, which is a (Nl,Nmono_cross) ndarray, and the 
      quadratic magnification contribution Cgg_mag_quad, which is a
      (Nl) ndarray. Given a set of monomial coefficients "coeff_a" and 
      "coeff_x" for the auto and cross respectively, the full prediction is: 
        Cgg = np.dot(Cgg_clust,coeff_a) + (5*s-2) * np.dot(Cgg_mag_lin,coeff_x) 
              + (5*s-2)**2 * Cgg_mag_quad
      where s is the slop of the cumulative magnitude func.  

      Parameters
      ----------
      i: int
         index of galaxy sample, i.e. i = 0,1,...,Ng-1
      thy_args: type can vary according to theory codes
         cosmological inputs
      lmax: int, default=500
         maximum multipole (inclusive)
      """
      # Compute power spectra (if necessary) by 
      # setting thy_args. The "kgrid" is defined
      # such that kgrid[i,j] = (l[j]+0.5)/chi[i]
      self.thy_args = thy_args
      l          = np.arange(lmax+1) ; Nl = len(l)
      kgrid = (np.tile(l+0.5,self.Nz)/np.repeat(self.chi,Nl)).reshape((self.Nz,Nl))
      
      Wg_clust   = self.Wg_clust[:,i]           # (Nz) ndarray
      Wg_mag     = self.Wg_mag[:,i]             # (Nz) ndarray
      kgg        = self.Pgg_eval[i][:,0]        # assume that k = fist col
      Pgg        = self.Pgg_eval[i][:,1:]       # monomials = (Nk,Nmono_auto) ndarray
      kgm        = self.Pgm_eval[i][:,0]        # assume that k = fist col
      Pgm        = self.Pgm_eval[i][:,1:]       # monomials = (Nk,Nmono_cros) ndarray
      Nmono_auto = Pgg.shape[1]                 # number of monomials for auto
      Nmono_cros = Pgm.shape[1]                 # number of monomials for cross
      
      # "clustering contribution"
      Cgg_clust = np.zeros((Nl,Nmono_auto))
      for j in range(Nmono_auto):
         pint = Spline(kgg,Pgg[:,j],ext=1)
         Cgg_clust[:,j] = limberIntegral(self.chi, Wg_clust, Wg_clust, pint(kgrid))
         
      # (linear) magnification contribution
      Cgg_mag_lin = np.zeros((Nl,Nmono_cros))
      for j in range(Nmono_cros):
         pint = Spline(kgm,Pgm[:,j],ext=1)
         Cgg_mag_lin[:,j] = 2.*limberIntegral(self.chi, Wg_clust, Wg_mag, pint(kgrid))
       
      # (quadratic) magnification contribution
      k       = self.Pmm_eval[:,0]          # assume that fist col. = k
      Pmm     = self.Pmm_eval[:,1:]         # (Nk,Nz) ndarray
      Pgrid   = np.zeros((self.Nz,Nl))
      for j in range(self.Nz):
         Pgrid[j,:] = Spline(k,Pmm[:,j],ext=1)(kgrid[j,:])
      Cgg_mag_quad = limberIntegral(self.chi, Wg_mag, Wg_mag, Pgrid)
      
      return Cgg_clust, Cgg_mag_lin, Cgg_mag_quad
