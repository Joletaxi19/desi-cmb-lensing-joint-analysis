import jax.numpy as np
from   jax import jit
from   jax.scipy.interpolate import RegularGridInterpolator as RGI

from numpy             import loadtxt
from scipy.integrate   import simps
from scipy.interpolate import interp1d
from datetime          import datetime
from functools         import partial

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

@jit   
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
   
@jit
def interpGrid(ptable,kgrid):
   """
   Interpolates a set of power spectrum monomials
   (ptable) onto a grid of wavenumbers (kgrid).
   
   Parameters
   ----------
   ptable: (Nk,1+Nmono) ndarray
      power spectrum table with Nmono monomials. The 
      first column is assumed to be the wavenumbers k.
   krid: (any shape) ndarray
      grid of wavenumbers to interpolate the table on
      
   Returns ndarray with shape = kgrid.shape + (Nmono,)
   """
   k       = ptable[:,0]              # (Nk)       ndarray
   pt      = ptable[:,1:]             # (Nk,Nmono) ndarray
   mono    = np.arange(pt.shape[1])   # index monomials
   interp  = RGI((k,mono), pt, fill_value=0., bounds_error=False, method='linear')
   KG,MONO = np.meshgrid(kgrid.flatten(),mono,indexing='ij')
   newArr  = np.zeros(KG.shape+(2,))
   newArr  = newArr.at[:,:,0].set(KG)
   newArr  = newArr.at[:,:,1].set(MONO)
   res     = interp(newArr).reshape(kgrid.shape+(pt.shape[1],))
   return res
   
####################################
##
## A class for computing Cgg and Ckg
##
####################################

class limb():
   """
   A class for calculating Cgg and Ckg within the Limber
   approximation given a set of redshift distributions 
   and codes for computing Pgm, Pgg, Pmm, and the background 
   cosmology. Throughout we use the 'effective redshift' 
   approximation when evaluating integrals over Pgg or Pgm. 
   We include the evolution of Pmm with redshift when 
   calculating magnification contributions.


   Attributes
   ----------
   Ng: int
      number of galaxy samples
   Nz: int
      number of redshifts used for Limber integrals
   Nl: int
      number of multipoles 
   z: (Nz) ndarray
      redshifts used for Limber integrals
   l: (Nl) ndarray
      multipoles used when computing Cl's
   dNdz: (Nz,Ng) ndarray
      redshift distributions of the galaxy samples 
      normalized so that \int dN/dz dz = 1
   thy_fid: type can vary according to theory codes
      fiducial cosmological inputs 
   zeff: (Ng) ndarray
      effective redshifts of the galaxy samples for 
      a specific fiducial cosmology
   
   Methods
   -------
   Pgm(thy_args,z)
      returns galaxy-matter cross-spectrum table at
      redshift z
   Pgg(thy_args,z)
      returns galaxy power spectrum table at redshift z
   Pmm(thy_args,zs)
      returns table of matter power spectra evaluated 
      at redshifts zs
   background(thy_args,zs)
      returns OmM,chistar,Ez,chi, where OmM and chistar
      are floats, while Ez and chi are evaluated at 
      redshifts zs
   gridNg(x)
      places (Ng) ndarray x on a (Nz,Ng) grid by 
      repeating x Nz times
   gridNz
      places (Nz) ndarray x on a (Nz,Ng) grid by 
      repeating x Ng times
   computeZeff(thy_args)
      computes the effective redshifts and saves them 
      to self.zeff
   projectionKernels(thy_args)
      returns the projection kernels for CMB lensing
      and galaxies (returns clustering and magnification
      contributions separately)
   evaluate
      returns projections kernels and power spectra 
      (after being evaluated at the appropriate redshifts)
   computeCkg
      computes the galaxy-convergence cross power. Returns
      the magnification contribution separately.
   computeCgg
      computes the galaxy power spectrum. Returns linear
      and quadratic magnification contributions separately.
   """
   
   def __init__(self, dNdz_fname, thy_fid, Pgm, Pgg, Pmm, background, lmax=500, l=None, zmin=0.001, zmax=2., Nz=50):
      """
      Parameters
      ----------
      dNdz_fname: str
         redshift distribution filename. Should load to a (Z,1+Ng) 
         ndarray, i.e. the redshift distributions (dN/dz) of the 
         Ng galaxy samples, with the first column being the Z redshifts.
      thy_fid: type can vary according to theory codes
         fiducial cosmological inputs. Used to compute the effective
         redshifts.
      Pgm: method
         takes (thy_args,z) as an input, where z is a float. Returns
         (Nk,1+Nmono) ndarray, where the first column is k, and the 
         remaining Nmono columns are (galaxy cross matter) power 
         spectrum monomials.
      Pgg: method
         takes (thy_args,z) as an input, where z is a float. Returns
         (Nk,1+Nmono) ndarray, where the first column is k, and the 
         remaining Nmono columns are (galaxy auto) power spectrum 
         monomials.
      Pmm: method
         takes (thy_args,z) as an input, where z is a (Z) ndarray. 
         Returns (Nk,1+Z) ndarray, where the first column is k, 
         and the remaining Z columns are the matter power spectrum
         evaluated at each redshift.
      background: method
         takes (thy_args, z) as an input, where z is a (Z) ndarray.
         Returns OmM [float], the distance to the surface of last 
         scatter [float], E(z) = H(z)/H0 [(Z) ndarray] evaluated
         on the input list of redshifts, as well as the comoving 
         distance [(Z) ndarray] evaluated on the input redshifts. 
      lmax: int, default = 500
         maximum multipole used when computing Cl's. If l is not 
         specified l = np.arange(lmax+1).
      l: (Nl) ndarray
         multipoles used when computing Cl's
      zmin: float, default = 0.001
         minimum redshift used for computing Limber integrals
      zmax: float, default =2.
         maximum redshift used for computing Limber integrals
      Nz: int, default = 50
         number of redshifts used for computing Limber integrals
      """
      dNdz       = loadtxt(dNdz_fname)
      Ng         = dNdz.shape[1] - 1
      self.Ng    = Ng
      self.Nz    = Nz
      self.z     = np.linspace(zmin,zmax,Nz)
      if l is not None: self.l = l
      else:             self.l = np.arange(lmax+1) 
      self.Nl    = len(self.l)
      # some helper functions
      def gridNg(x): return np.tile(x,Nz).reshape((Nz,Ng))
      def gridNz(x): return np.repeat(x,Ng).reshape((Nz,Ng))
      self.gridNg = jit(gridNg)
      self.gridNz = jit(gridNz)
      # evaluate dNdz on regular grid and normalize it so 
      # that \int dN/dz dz = 1 for each galaxy sample
      self.dNdz  = interp1d(dNdz[:,0],dNdz[:,1:],axis=0,bounds_error=False,fill_value=0.)(self.z)
      self.dNdz  = self.dNdz.reshape((self.Nz,self.Ng))   # does nothing for Ng > 1
      norm       = simps(self.dNdz, x=self.z, axis=0)     # (Ng) ndarray
      norm       = self.gridNg(norm)
      self.dNdz /= norm
      # jit and store theory predictions
      self.Pgm               = jit(Pgm)
      self.Pgg               = jit(Pgg)
      self.Pmm               = jit(Pmm)
      self.background        = jit(background)
      # store fiducial cosmology and compute effective redshifts
      self.thy_fid  = thy_fid
      self.computeZeff()
      print('Effective redshifts of the',Ng,'galaxy samples:')
      for i in range(self.Ng): print(f'z{i} =','%.2f'%self.zeff[i])
      print('\n')
      print('Doing JIT compilations')
      print('----------------------')
      self.Pgm(thy_fid,1.)            ; print(datetime.now().strftime("%H:%M:%S"),': Pgm compiled')
      self.Pgg(thy_fid,1.)            ; print(datetime.now().strftime("%H:%M:%S"),': Pgg compiled')
      self.Pmm(thy_fid,self.z)        ; print(datetime.now().strftime("%H:%M:%S"),': Pmm compiled')
      self.background(thy_fid,self.z) ; print(datetime.now().strftime("%H:%M:%S"),': background compiled')
      self.projectionKernels(thy_fid) ; print(datetime.now().strftime("%H:%M:%S"),': projectionKernels compiled')
      self.evaluate(thy_fid)          ; print(datetime.now().strftime("%H:%M:%S"),': evaluate compiled')
      self.computeCkg(0,thy_fid)      ; print(datetime.now().strftime("%H:%M:%S"),': computeCkg compiled')
      self.computeCgg(0,thy_fid)      ; print(datetime.now().strftime("%H:%M:%S"),': computeCgg compiled')
      print('----------------------')
      print('Finished compilations, ready to calculate!')


   def computeZeff(self):
      """
      Computes the effective redshift for each galaxy sample 
      assuming the fiducial cosmology and saves them to 
      self.zeff, which is a (Ng) ndarray.
      """
      OmM,chistar,Ez,chi = self.background(self.thy_fid,self.z) 
      _,Wg,_             = self.projectionKernels(self.thy_fid) 
      zeff = lambda i: effRedshift(chi, Wg[:,i], Wg[:,i], self.z)
      self.zeff = np.array([zeff(i) for i in range(self.Ng)])

    
   @partial(jit, static_argnums=(0,))           
   def projectionKernels(self, thy_args):
      """
      Computes the projection kernels [h/Mpc] for CMB lensing 
      and each galaxy sample. The CMB lensing kernel (Wk) is 
      a (Nz) ndarray. The galaxy kernels are (Nz,Ng) ndarrays. 
      The full galaxy kernel is 
               Wg = Wg_clust + (5*s-2) * Wg_mag
      where s is the slope of the cumulative magnitude func. 
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
      """
         
      OmM,chistar,Ez,chi = self.background(thy_args,self.z)
      H0                 = 100./299792.458 # [h/Mpc] units
      ## CMB lensing
      Wk  = 1.5*OmM*H0**2.*(1.+self.z)
      Wk *= chi*(chistar-chi)/chistar
      ## Galaxies
      # clustering contribution
      Wg_clust  = self.gridNz(H0*Ez) * self.dNdz  
      # magnification bias contribution
      def integrate_z_zstar(x):
         # approximates the integral 
         # \int_z^{zstar} dz' x(z') 
         # with a Riemann sum
         x = np.flip(x,axis=0)
         x = np.cumsum(x,axis=0) * (self.z[1]-self.z[0])
         return np.flip(x,axis=0)
      Wg_mag  = self.gridNz(chi)*integrate_z_zstar(self.dNdz)
      Wg_mag -= self.gridNz(chi**2)*integrate_z_zstar(self.gridNz(1./chi)*self.dNdz)
      Wg_mag *= self.gridNz(1.5*OmM*H0**2.*(1.+self.z))
      
      return Wk,Wg_clust,Wg_mag

    
   @partial(jit, static_argnums=(0,))
   def evaluate(self, thy_args):
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

      Nmono is the number of monomials (e.g. 1, b1, b2, ...), which can in 
      general be different for Pgm and Pgg. The "+1" is a column of ks.
      
      Parameters
      ----------
      thy_args: type can vary according to theory codes
         cosmological inputs
      """
      _,_,_,chi          = self.background(thy_args,self.z)
      Wk,Wg_clust,Wg_mag = self.projectionKernels(thy_args)
      Pgm_eval = np.array([self.Pgm(thy_args,z) for z in self.zeff])
      Pgg_eval = np.array([self.Pgg(thy_args,z) for z in self.zeff])
      Pmm_eval = self.Pmm(thy_args,self.z)
      return chi,Wk,Wg_clust,Wg_mag,Pgm_eval,Pgg_eval,Pmm_eval

    
   @partial(jit, static_argnums=(0,))  
   def computeCkg(self, i, thy_args):
      """
      Computes the CMB lensing-galaxy cross-correlation
      within the Limber approximation. Returns a clustering 
      contribution Ckg_clust in the form of a "monomial table", 
      i.e. a (Nl,Nmono) ndarray, as well as the magnification 
      contribution Ckg_mag, which is a (Nl) ndarray. Given a 
      set of monomial coefficients "coeff", which is a (Nmono) 
      ndarray, the full prediction is: 
        Ckg = np.dot(Ckg_clust,coeff) + (5*s-2) * Ckg_mag
      where s is the slope of the cumulative magnitude func.  

      Parameters
      ----------
      i: int
         index of galaxy sample, i.e. i = 0,1,...,Ng-1
      thy_args: type can vary according to theory codes
         cosmological inputs
      """
      # Evaluate projection kernels and power spectra.
      # The "kgrid" is defined such that kgrid[i,j] = (l[j]+0.5)/chi(z[i])
      chi,Wk,Wg_clust,Wg_mag,Pgm_eval,Pgg_eval,Pmm_eval = self.evaluate(thy_args)                 
      kgrid = (np.tile(self.l+0.5,self.Nz)/np.repeat(chi,self.Nl)).reshape((self.Nz,self.Nl))
       
      # "clustering contribution"
      PgmTable  = Pgm_eval[i]                # (Nk,1+Nmono) ndarray
      Nmono     = PgmTable.shape[1] - 1 
      PgmIntrp  = interpGrid(PgmTable,kgrid) # kgrid.shape + (Nmono,) ndarray
      Ckg_clust = np.zeros((self.Nl,Nmono))
      for j in range(Nmono):
         Ckg_mono  = limberIntegral(chi, Wk, Wg_clust[:,i], PgmIntrp[:,:,j])
         Ckg_clust = Ckg_clust.at[:,j].set(Ckg_mono)
          
      # magnification contribution
      PmmIntrp = interpGrid(Pmm_eval,kgrid)                  # kgrid.shape + (Nz,) ndarray
      Pgrid    = np.diagonal(PmmIntrp,axis1=0,axis2=2).T     # kgrid.shape ndarray
      Ckg_mag  = limberIntegral(chi, Wk, Wg_mag[:,i], Pgrid)
      
      return Ckg_clust, Ckg_mag

    
   @partial(jit, static_argnums=(0,))    
   def computeCgg(self, i, thy_args):
      """
      Computes the galaxy auto-correlation within the Limber approximation. 
      Returns a clustering contribution Cgg_clust in the form of a "monomial 
      table", i.e. a (Nl,Nmono_auto) ndarray, the linear magnification contribution 
      Cgg_mag_lin, which is a (Nl,Nmono_cross) ndarray, and the quadratic 
      magnification contribution Cgg_mag_quad, which is a (Nl) ndarray. Given a set 
      of monomial coefficients "coeff_a" and "coeff_x" for the auto and cross 
      respectively, the full prediction is: 
                 Cgg = np.dot(Cgg_clust,coeff_a) 
                       + (5*s-2) * np.dot(Cgg_mag_lin,coeff_x) 
                       + (5*s-2)**2 * Cgg_mag_quad
      where s is the slope of the cumulative magnitude func.  

      Parameters
      ----------
      i: int
         index of galaxy sample, i.e. i = 0,1,...,Ng-1
      thy_args: type can vary according to theory codes
         cosmological inputs
      """
      # Evaluate projection kernels and power spectra.
      # The "kgrid" is defined such that kgrid[i,j] = (l[j]+0.5)/chi(z[i])
      chi,Wk,Wg_clust,Wg_mag,Pgm_eval,Pgg_eval,Pmm_eval = self.evaluate(thy_args)                 
      kgrid = (np.tile(self.l+0.5,self.Nz)/np.repeat(chi,self.Nl)).reshape((self.Nz,self.Nl))
      
      Wg_clust   = Wg_clust[:,i]        # (Nz) ndarray
      Wg_mag     = Wg_mag[:,i]          # (Nz) ndarray
      PggTable   = Pgg_eval[i]          # (Nk,1+Nmono_auto) ndarray
      PgmTable   = Pgm_eval[i]          # (Nk,1+Nmono_cros) ndarray
      Nmono_auto = PggTable.shape[1]-1  # number of monomials for auto
      Nmono_cros = PgmTable.shape[1]-1  # number of monomials for cross
      
      # "clustering contribution"
      PggIntrp  = interpGrid(PggTable,kgrid) # kgrid.shape + (Nmono_auto,) ndarray
      Cgg_clust = np.zeros((self.Nl,Nmono_auto))
      for j in range(Nmono_auto):
         Cgg_mono  = limberIntegral(chi, Wg_clust, Wg_clust, PggIntrp[:,:,j])
         Cgg_clust = Cgg_clust.at[:,j].set(Cgg_mono)
         
      # (linear) magnification contribution
      PgmIntrp    = interpGrid(PgmTable,kgrid) # kgrid.shape + (Nmono_cros,) ndarray
      Cgg_mag_lin = np.zeros((self.Nl,Nmono_cros))
      for j in range(Nmono_cros):
         Cgg_mono    = limberIntegral(chi, Wg_clust, Wg_mag, PgmIntrp[:,:,j])
         Cgg_mag_lin = Cgg_mag_lin.at[:,j].set(2.*Cgg_mono)

      # (quadratic) magnification contribution
      PmmIntrp     = interpGrid(Pmm_eval,kgrid)                 # kgrid.shape + (Nz,) ndarray
      Pgrid        = np.diagonal(PmmIntrp,axis1=0,axis2=2).T    # kgrid.shape ndarray
      Cgg_mag_quad = limberIntegral(chi, Wg_mag, Wg_mag, Pgrid)
      
      return Cgg_clust, Cgg_mag_lin, Cgg_mag_quad
