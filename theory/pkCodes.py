# This is a wrapper around various cosmology/PT/emulator codes. This file contains 
# several methods to compute real-space power spectra. Each method should have 
# (thy_args, z) as its first two arguments, and can optionally have trailing arguments. 
# These methods return power spectrum tables, which are then multiplied by biases, 
# counterterms, etc. to return the full theory prediction.
#
#
# Currently wrapped:
# - Pgg with velocileptors
# - Pgm with velocileptors
# - Pmm with HaloFit
# - Pmm with Aemulus emulator

# ingredients
import numpy as np
from classy import Class
from velocileptors.LPT.cleft_fftw import CLEFT

# fiducial k-grid [h/Mpc] on which we evaluate (Pgm,Pgg,Pmm) tables
ks = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.029),60, endpoint=True),\
                        np.arange(0.03,0.51,0.01),\
                        np.linspace(0.52,5.,20)) )

def getCosmo(thy_args):
   """
   Returns a CLASS object.

   Parameters
   ----------
   thy_args: dict
      cosmological inputs to CLASS
   """
   params = {'output': 'mPk','P_k_max_h/Mpc': 20.,'non linear':'halofit','z_pk': '0.0,20',
             'N_ur': 1.0196,'N_ncdm': 2,'m_ncdm': '0.01,0.05'}
   cosmo = Class()
   cosmo.set(params)
   cosmo.set(thy_args)
   cosmo.compute()
   return cosmo


def pmmHalofit(thy_args,z,k=None):
   """
   Returns a table with shape (Nk,2). The 
   first column is k, while the second column
   is the halofit prediction for the non-linear
   matter power spectrum.

   Parameters
   ----------
   thy_args: dict
      cosmological inputs to CLASS
   z: float OR ndarray
      redshift
   k: ndarray, optional
      wavevectors [h/Mpc] on which to evaluate
      the power spectrum table
   """
   if k is None: k = ks
   cosmo = getCosmo(thy_args)
   h = cosmo.h()
   Pk = lambda zz: np.array([cosmo.pk(kk*h,zz)*h**3 for kk in k])
   if isinstance(z, (np.floating, float)): return np.array([k,Pk(z)]).T
   else: res = [Pk(zz) for zz in z]
   return np.array([k]+res).T

   
def pggVelocileptors(thy_args,z,k=None):
   """
   Returns a Pgg table with shape (Nk,15).
   The first column is k, while the remaining 
   14 have monomial coefficients:
   [1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, 
   b2*bs, bs**2, b3, b1*b3, N0, alpha0]

   Uses pk_cb_lin as the input linear power
   spectrum. 

   Parameters
   ----------
   thy_args: dict
      cosmological inputs to CLASS
   z: float
      redshift
   k: ndarray, optional
      wavevectors [h/Mpc] on which to evaluate
      the power spectrum table
   """
   if k is None: k = ks
   cosmo = getCosmo(thy_args)
   h     = cosmo.h()
   klin  = np.logspace(-3,np.log10(20.),4000)
   plin  = np.array([cosmo.pk_cb_lin(kk*h,z)*h**3 for kk in klin])
   cleft = CLEFT(klin,plin,cutoff=5.)
   cleft.make_ptable(kmin=min(k),kmax=max(k),nk=len(k))
   kout,za    = cleft.pktable[:,0],cleft.pktable[:,13]   
   res        = np.zeros((len(kout),17))
   res[:,:13] = cleft.pktable[:,:13]
   res[:,13]  = np.ones_like(kout)
   res[:,14]  = -0.5*kout**2*za
   return res

   
def pgmVelocileptors(thy_args,z,k=None):
   """
   Returns a Pgm table with shape (Nk,7).
   The first column is k, while the remaining 
   6 have monomial coefficients:
   [1, b1, b2, bs, b3, alphaX]
   
   Uses sqrt(pk_cb_lin * pk_lin) as the input
   linear power spectrum to account for 
   neutrinos [2204.10392].

   Parameters
   ----------
   thy_args: dict
      cosmological inputs to CLASS
   z: float
      redshift
   k: ndarray, optional
      wavevectors [h/Mpc] on which to evaluate
      the power spectrum table
   """
   if k is None: k = ks
   cosmo = getCosmo(thy_args)
   h     = cosmo.h()
   klin  = np.logspace(-3,np.log10(20.),4000) # more ks are cheap
   plin  = np.array([cosmo.pk_cb_lin(kk*h,z)*h**3 for kk in klin])
   plin *= np.array([cosmo.pk_lin(kk*h,z)*h**3 for kk in klin])
   plin  = np.sqrt(plin)
   cleft = CLEFT(klin,plin,cutoff=5.)
   cleft.make_ptable(kmin=min(k),kmax=max(k),nk=len(k))
   kout,za  = cleft.pktable[:,0],cleft.pktable[:,13]
   res      = np.zeros((len(kout),7))
   res[:,0] = kout                     # k
   res[:,1] = cleft.pktable[:,1 ]      # 1
   res[:,2] = cleft.pktable[:,2 ]/2.   # b1
   res[:,3] = cleft.pktable[:,4 ]/2.   # b2
   res[:,4] = cleft.pktable[:,7 ]/2.   # bs
   res[:,5] = cleft.pktable[:,11]/2.   # b3
   res[:,6] = -0.5*kout**2*za          # alphaX
   return res
   
   
import jax.numpy as jnp
# OBVIOUSLY NEED TO REMOVE HARD-CODED PATH
# IN FAVOR OF PIP INSTALLATION. HOWEVER, 
# THE CURRENT GITHUB DOESN'T HAVE NN WEIGHTS
# AND ALSO IS NOT JAX-COMPATIBLE, WHICH IS WHY I'm 
# LOADING MY OWN REPO (GOT WEIGHTS FROM JOE, AND 
# MODIFIED heft_emu -> heft_emu_jax)
import sys
sys.path.append('/home/noah/Berkeley/aemulus_heft') # running on my laptop since perlmutter is a pos
from aemulus_heft.heft_emu_jax import NNHEFTEmulator

nnemu = NNHEFTEmulator()

def pmmHEFT(thy_args,z):
   """
   """
   omb,omc,ns,As,H0 = thy_args
   #cosmo = jnp.array([[omb, omc, -1., ns, As, H0, 0.06, zz] for zz in z])
   cosmo = jnp.zeros((len(z),8))
   cosmo = cosmo.at[:,-1].set(z)
   cosmo = cosmo.at[:,:-1].set([omb, omc, -1., ns, As, H0, 0.06])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   res = jnp.zeros((len(k_nn),len(z)+1))
   res = res.at[:,0].set(k_nn)
   #for i in range(len(z)): res = res.at[:,i+1].set(spec_heft_nn[i,0,:])
   res = res.at[:,1:].set(jnp.swapaxes(spec_heft_nn[:,0,:],0,1))
   return res

def ptableHEFT(thy_args,z):
   """
   CURRENTLY RETURNING ALL MONOMIALS, NOT APPROPRIATE FOR
   PGM OR PMM, BUT GOOD ENOUGH FOR TESTING CODE
   """
   omb,omc,ns,As,H0 = thy_args
   cosmo = jnp.atleast_2d([omb, omc, -1., ns, As, H0, 0.06, z])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   Nmono = spec_heft_nn.shape[1]
   res = jnp.zeros((len(k_nn),Nmono+1))
   res = res.at[:,0].set(k_nn)
   for i in range(Nmono): res = res.at[:,i+1].set(spec_heft_nn[0,i,:])
   return res