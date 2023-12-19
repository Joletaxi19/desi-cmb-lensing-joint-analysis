import numpy as np
from classy import Class
import sys
import os
from time import sleep
from scipy.interpolate import interp1d

def get_sigma8(omb,omc,ns,As,H0,Mnu,counter=0):
    params = {'output': 'mPk','A_s': 1e-10*np.exp(As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except:
        if counter > 50:
            print('Ran CLASS 50 times with params')
            print('omb =',omb)
            print('omc =',omc)
            print('As  =',As)
            print('ns  =',ns)
            print('H0  =',H0)
            print('Mnu =',Mnu)
            print('and couldnt get a reasonable sigma8 value, returning 0.75')
            return 0.75
        return get_sigma8(omb,omc,ns,As,H0,Mnu,counter=counter+1)
    return cosmo.sigma8()

def get_OmM_classy(omb,omc,ns,As,H0,Mnu):
    params = {'output': 'mPk','z_pk': '0.0,1','A_s': 1e-10*np.exp(As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    return cosmo.Omega0_m()

"""
def get_OmM(omb,omc,ns,As,H0,Mnu):
    params = {'output': 'mPk','z_pk': '0.0,1','A_s': 1e-10*np.exp(As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    return cosmo.Omega0_m()

def get_sigma8_hacky(omb,omc,ns,As,H0,Mnu):
    pmm = pmmHEFT(np.array([omb,omc,ns,As,H0,Mnu]),np.array([0.]))
    k = pmm[:,0]
    p = pmm[:,1]
    s8_proxy = simps((3*jn(1,8*k)/(8*k))**2*k**2*p/2/np.pi**2, x=k)**0.5
"""    
    
## newer and betterer code  
    
    
def get_H0(OmMh3, omega_cdm, omega_b, m_ncdm): return 100*np.real(np.roots([m_ncdm/41.844,0,omega_cdm+omega_b,-OmMh3])[-1])

def get_OmM(OmMh3,H0): return OmMh3/(H0/100)**3
    
def get_sigma8_classy(omega_b,omega_cdm,n_s,ln1e10As,OmMh3,m_ncdm):
    H0 = get_H0(OmMh3, omega_cdm, omega_b, m_ncdm)
    params = {'output': 'mPk','z_pk': '0.0,20','A_s': 1e-10*np.exp(ln1e10As),'n_s': n_s,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': m_ncdm,'tau_reio': 0.0568,
             'omega_b': omega_b,'omega_cdm': omega_cdm}
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo.sigma8()

def sigma8_emu_fname(OmMh3, omega_b, m_ncdm, ln1e10As, n_s):
    return f'OmMh3-{OmMh3}_omega_b-{omega_b}_m_ncdm-{m_ncdm}_ln1e10As-{ln1e10As}_n_s-{n_s}.txt'

def train_sigma8_emu(OmMh3, omega_b, m_ncdm, n_s, ln1e10As=3.,omc_min=0.08,omc_max=0.16,N_omc=50):
    sigma8 = lambda omega_cdm: get_sigma8_classy(omega_b,omega_cdm,n_s,ln1e10As,OmMh3,m_ncdm)
    omega_cdms = np.linspace(omc_min,omc_max,N_omc)
    sigma8s    = np.array([sigma8(omc) for omc in omega_cdms])
    fname      = 'sigma8_emus/'+sigma8_emu_fname(OmMh3,omega_b,m_ncdm,ln1e10As,n_s)
    np.savetxt(fname,np.array([omega_cdms,sigma8s]).T,header='Columns are: omega_cdm, sigma8')
    
def get_sigma8_emu(omega_b,omega_cdm,n_s,ln1e10As,OmMh3,m_ncdm):
    ln1e10As_fid=3.0
    fname = 'sigma8_emus/'+sigma8_emu_fname(OmMh3,omega_b,m_ncdm,ln1e10As_fid,n_s)
    if not os.path.exists(fname):
        print('Training new emulator...',flush=True)
        train_sigma8_emu(OmMh3, omega_b, m_ncdm, n_s, ln1e10As=ln1e10As_fid)
        sleep(15)
        print('finished training!',flush=True)
    dat = np.loadtxt(fname)
    sigma8_interp = interp1d(dat[:,0],dat[:,1],kind='cubic')
    return sigma8_interp(omega_cdm) * (np.exp(ln1e10As)/np.exp(ln1e10As_fid))**0.5 
    