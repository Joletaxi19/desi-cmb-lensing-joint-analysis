import numpy as np
from scipy.integrate   import simps
<<<<<<< HEAD
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

=======
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

>>>>>>> 42ed741 (mask making scripts)
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
   
<<<<<<< HEAD
   def __init__(self, dNdz_fname, zmin=0.001, zmax=5., Nz=500,
                      Pgm=None, Pgg=None, Pmm=None, 
                      background=None, thy_fid={}):
=======
   def __init__(self, dNdz_fname, thy_fid, Pgm, Pgg, Pmm, background, lmax=1000, Nlval=64, zmin=0.001, zmax=2., Nz=50):
>>>>>>> 42ed741 (mask making scripts)
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
<<<<<<< HEAD
      # evaluate dNdz on regular grid and normalize it so 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = interp1d(dNdz[:,0],dNdz[:,1:],axis=0,bounds_error=False,fill_value=0.)(self.z)
      self.dNdz  = self.dNdz.reshape((self.Nz,self.Ng))   # does nothing for Ng > 1
=======
      self.l     = np.arange(lmax+1) 
      self.lval  = np.logspace(0,np.log10(lmax),Nlval)
      self.Nl    = len(self.l)
      self.Nlval = len(self.lval)
      # evaluate dNdz on regular grid and normalize it such 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = np.zeros((self.Nz,self.Ng))
      for j in range(self.Ng): self.dNdz[:,j] = Spline(dNdz[:,0],dNdz[:,j+1],ext=1)(self.z)
>>>>>>> 42ed741 (mask making scripts)
      norm       = simps(self.dNdz, x=self.z, axis=0)     # (Ng) ndarray
      norm       = self.gridMe(norm)
      self.dNdz /= norm
      # store theory predictions 
      self.Pgm         = Pgm
      self.Pgg         = Pgg
      self.Pmm         = Pmm
<<<<<<< HEAD
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

    
=======
      self.background = background
      # store fiducial cosmology and compute effective redshifts
      self.thy_fid  = thy_fid                   
      self.computeZeff()
                              
>>>>>>> 42ed741 (mask making scripts)
   def computeZeff(self):
      """
      Computes the effective redshift for each galaxy sample 
      assuming the fiducial cosmology and saves them to 
<<<<<<< HEAD
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
=======
      self.zeff, which is a (Ng) ndarray.
      """
      OmM,chistar,Ez,chi = self.background(self.thy_fid,self.z)
      _,Wg,_             = self.projectionKernels(self.thy_fid)
      zeff = lambda i: simps(Wg[:,i]**2*self.z/chi**2,x=chi)/simps(Wg[:,i]**2/chi**2,x=chi)
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
>>>>>>> 42ed741 (mask making scripts)
      general be different for Pgm and Pgg. The "+1" is a column of ks.
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
<<<<<<< HEAD
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
=======
      """
      OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      Wk,Wg_clust,Wg_mag = self.projectionKernels(thy_args,bkgrnd=[OmM,chistar,Ez,chi])
      Pgm_eval = self.Pgm(thy_args,self.zeff[i])
      Pgg_eval = self.Pgg(thy_args,self.zeff[i])
      Pmm_eval = self.Pmm(thy_args,self.z)
      return chi,Wk,Wg_clust,Wg_mag,Pgm_eval,Pgg_eval,Pmm_eval
>>>>>>> 42ed741 (mask making scripts)

       
   def gridMe(self,x):
      """
      Places input on a (Nz,Ng) grid. If x is z-independent, 
      repeats Ng times. If x is galaxy-independent, repeats 
      Nz times. If x is a float, repeats Nz*Ng times. 
      
      Parameters
      ----------
      x: float, (Nz) ndarray, OR (Ng) ndarray
         the input to be gridded
<<<<<<< HEAD
         
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
=======
      """
      if isinstance(x,float):
         return x*np.ones_like(self.dNdz)
>>>>>>> 42ed741 (mask making scripts)
      N = x.shape[0]
      if N == self.Ng:
         return np.tile(x,self.Nz).reshape((self.Nz,self.Ng))
      if N == self.Nz:
         return np.repeat(x,self.Ng).reshape((self.Nz,self.Ng))
<<<<<<< HEAD
      else: 
         s = 'input must satisfy len = self.Ng or self.Nz'
         raise RuntimeError(s)
      
      
   def projectionKernels(self, thy_args):
=======
      
      
   def projectionKernels(self, thy_args, bkgrnd=None):
>>>>>>> 42ed741 (mask making scripts)
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
<<<<<<< HEAD
      if self.background is None:
         s  = 'must provide a background code to compute projection kernels'
         raise RuntimeError(s)
         
      OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      H0                 = 100./299792.458 # [h/Mpc] units
=======
      if bkgrnd is None: OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      else:              OmM,chistar,Ez,chi = bkgrnd
      H0 = 100./299792.458 # [h/Mpc] units
>>>>>>> 42ed741 (mask making scripts)
      ## CMB lensing
      Wk  = 1.5*OmM*H0**2.*(1.+self.z)
      Wk *= chi*(chistar-chi)/chistar
      ## Galaxies
      # clustering contribution
      Wg_clust  = self.gridMe(H0*Ez) * self.dNdz  
      # magnification bias contribution
      def integrate_z_zstar(x):
<<<<<<< HEAD
         # approximates the integral 
         # \int_z^{zstar} dz' x(z') 
         # with a Riemann sum
=======
         # approximates \int_z^{zstar} dz' x(z') with a Riemann sum
>>>>>>> 42ed741 (mask making scripts)
         x = np.flip(x,axis=0)
         x = np.cumsum(x,axis=0) * (self.z[1]-self.z[0])
         return np.flip(x,axis=0)
      Wg_mag  = self.gridMe(chi)*integrate_z_zstar(self.dNdz)
      Wg_mag -= self.gridMe(chi**2)*integrate_z_zstar(self.gridMe(1./chi)*self.dNdz)
      Wg_mag *= self.gridMe(1.5*OmM*H0**2.*(1.+self.z))
      
      return Wk,Wg_clust,Wg_mag
<<<<<<< HEAD
  
  
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
=======


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
>>>>>>> 42ed741 (mask making scripts)
