# ingredients
import numpy as np
from classy import Class
from velocileptors.LPT.cleft_fftw import CLEFT
from aemulus_heft.heft_emu import NNHEFTEmulator

# fiducial k-grid [h/Mpc] on which we evaluate (Pgm,Pgg) tables for velocileptors
ks = np.concatenate(([0.0005,],\
                     np.logspace(np.log10(0.0015),np.log10(0.029),60,endpoint=True),\
                     np.arange(0.03,0.51,0.01),\
                     np.linspace(0.52,5.,20)))
# aemulus nu HEFT emulator
nnemu = NNHEFTEmulator()

def getCosmo(thy_args):
   """
   Returns a CLASS object (with perturbations computed) for the cosmology
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]       
   params = {'output': 'mPk','P_k_max_h/Mpc': 20.,'non linear':'halofit','z_pk': '0.0,20',
             'A_s': 1e-10*np.exp(ln10As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   cosmo = Class()
   cosmo.set(params)
   cosmo.compute()
   return cosmo

# These are thin wrappers around various cosmology/PT/emulator codes for
# real-space power spectra. Each method should have (thy_args, z) as its 
# arguments. 
#
# Currently wrapped:
# - Pgg with Halofit, velocileptors, Aemulus emulator
# - Pgm with Hallofit velocileptors, Aemulus emulator
# - Pmm with HaloFit, Aemulus emulator

def pmmHalofit(thy_args,z):
   """
   Returns a (Nk,Nz+1) ndarray. The first column is k, while the remaining
   columns are the non-linear matter power spectrum evaluated at each z.

   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   z is a (Nz) ndarray
   """
   k = np.logspace(np.log10(0.005),np.log10(5.),200)
   cosmo = getCosmo(thy_args)
   h = cosmo.h()
   Pk = lambda zz: np.array([cosmo.pk(kk*h,zz)*h**3 for kk in k])
   if isinstance(z, (np.floating, float)): return np.array([k,Pk(z)]).T
   else: res = [Pk(zz) for zz in z]
   return np.array([k]+res).T

def pggHalofit(thy_args,z):
   """
   Returns a (Nk,2) ndarray. The first column is k, while the second 
   column is b_1^2 * P_{cb, halofit}.

   omb,omc,ns,ln10As,H0,Mnu,b1 = thy_args[:7]
   z is a float
   """
   k = np.logspace(np.log10(0.005),np.log10(5.),200)
   cosmo = getCosmo(thy_args)
   h = cosmo.h()
   Pk = lambda zz: np.array([cosmo.pk_cb(kk*h,zz)*h**3 for kk in k])
   return np.array([k,thy_args[6]**2*Pk(z)]).T

def pgmHalofit(thy_args,z):
   """
   Returns a (Nk,2) ndarray. The first column is k, while the second 
   column is b_1 * P_{cb, halofit}.

   omb,omc,ns,ln10As,H0,Mnu,b1 = thy_args[:7]
   z is a float
   """
   k = np.logspace(np.log10(0.005),np.log10(5.),200)
   cosmo = getCosmo(thy_args)
   h = cosmo.h()
   Pk = lambda zz: np.array([cosmo.pk_cb(kk*h,zz)*h**3 for kk in k])
   return np.array([k,thy_args[6]*Pk(z)]).T

def pggVelocileptors(thy_args,z,k=None):
   """
   Returns a Pgg table with shape (Nk,3).
   The first column is k, while the remaining 
   2 have monomial coefficients [1,alpha_auto].
   Uses pk_cb_lin as the input linear power
   spectrum. 

   Parameters
   ----------
   thy_args: list or ndarray
      omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   z: float
      redshift
   k: ndarray, optional
      wavevectors [h/Mpc] on which to evaluate
      the power spectrum table
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   if k is None: k = ks
   cosmo = getCosmo(thy_args)
   h     = cosmo.h()
   klin  = np.logspace(-3,np.log10(20.),4000)
   plin  = np.array([cosmo.pk_cb_lin(kk*h,z)*h**3 for kk in klin])
   cleft = CLEFT(klin,plin,cutoff=5.)
   cleft.make_ptable(kmin=min(k),kmax=max(k),nk=len(k))
   kout,za  = cleft.pktable[:,0],cleft.pktable[:,13]   
   res      = np.zeros((len(kout),3))
   res[:,0] = kout
   res[:,1] = cleft.combine_bias_terms_pk(b1, b2, bs, 0., 0., 0.)[1]
   res[:,2] = -0.5*kout**2*za
   return res

def pgmVelocileptors(thy_args,z,k=None):
   """
   Returns a Pgm table with shape (Nk,3).
   The first column is k, while the remaining 
   2 have monomial coefficients [1, alpha_cross].
   Uses sqrt(pk_cb_lin * pk_lin) as the input
   linear power spectrum [arXiv:2204.10392].

   Parameters
   ----------
   thy_args: list or ndarray
      omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   z: float
      redshift
   k: ndarray, optional
      wavevectors [h/Mpc] on which to evaluate
      the power spectrum table
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
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
   res      = np.zeros((len(kout),3))
   res[:,0] = kout
   res[:,1] = cleft.combine_bias_terms_pk_crossmatter(b1, b2, bs, 0., 0.)[1]
   res[:,2] = -0.5*kout**2*za
   return res

def pmmHEFT(thy_args,z):
   """
   Assumes thy_args[:5] = [omb,omc,ns,ln10As,H0,Mnu] and z = ndarray
   
   Returns res = (Nk,1+Nz) table where the first column is k and the
   remaining Nz columns are the matter power spectrum evaluated
   at each z.
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   Mnu          = max(Mnu,0.01) # HEFT is only valid for 0.01 < Mnu < 0.5 
   cosmo        = np.zeros((len(z),8))
   cosmo[:,-1]  = z 
   cosmo[:,:-1] = np.array([omb, omc, -1., ns, np.exp(ln10As)/10., H0, Mnu])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   res       = np.zeros((len(k_nn),len(z)+1))
   res[:,0]  = k_nn
   res[:,1:] = np.swapaxes(spec_heft_nn[:,0,:],0,1)
   return res

def ptableHEFT(thy_args,z):
   """
   Assumes thy_args[:5] = [omb,omc,ns,ln10As,H0,Mnu] and z = float
   
   Returns monomial table = (Nk,1+Nmono) ndarray. The first column is k, 
   while the order of the 15 monomials is:
   
   1-1, 1-cb, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb, 
   delta2-delta, delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2.
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   Mnu   = max(Mnu,0.01) # HEFT is only valid for 0.01 < Mnu < 0.5 
   cosmo = np.atleast_2d([omb, omc, -1., ns, np.exp(ln10As)/10., H0, Mnu, z])
   k_nn, spec_heft_nn = nnemu.predict(cosmo)
   Nmono     = spec_heft_nn.shape[1]
   res       = np.zeros((len(k_nn),Nmono+1))
   res[:,0]  = k_nn
   res[:,1:] = np.swapaxes(spec_heft_nn[0,:,:],0,1)
   return res
   
def pgmHEFT(thy_args,z):
   """
   Assumes thy_args = [omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs] and z = float
   
   Returns res = (Nk,3) ndarray, where the first column is k, the second column
   is the "bias contribution" (i.e. terms that cannot be analytically 
   marginalized over), while the third column is -0.5*k^2 P_{cb, 1}
   
   The full prediction is res[:,1] + alpha_x * res[:,2]
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   bterms_gm = np.array([0, 1, 0, b1, 0, 0, 0.5*b2, 0, 0, 0, bs, 0, 0, 0, 0])
   T         = ptableHEFT(thy_args,z)
   res       = np.zeros((T.shape[0],3))
   res[:,0]  = T[:,0]
   res[:,1]  = np.dot(T[:,1:],bterms_gm) # bias-contribution
   res[:,2]  = -0.5 * T[:,0]**2 * T[:,1] # counterterm
   return res  
  
def pggHEFT(thy_args,z):
   """
   Assumes thy_args = [omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs] and z = float
   
   Returns res = (Nk,3) ndarray, where the first column is k, the second column
   is the "bias contribution" (i.e. terms that cannot be analytically 
   marginalized over), while the third column is -0.5*k^2 P_{cb, cb}
   
   The full prediction is res[:,1] + alpha_a * res[:,2]
   """
   omb,omc,ns,ln10As,H0,Mnu,b1,b2,bs = thy_args
   bterms_gg = np.array([0, 0, 1, 0, 2*b1, b1**2, 0, b2, b2*b1, 0.25*b2**2, 0, 2*bs, 2*bs*b1, bs*b2, bs**2])
   T         = ptableHEFT(thy_args,z)
   res       = np.zeros((T.shape[0],3))
   res[:,0]  = T[:,0]
   res[:,1]  = np.dot(T[:,1:],bterms_gg) # bias-contribution
   res[:,2]  = -0.5 * T[:,0]**2 * T[:,2] # counterterm
   return res  