# Code to create mock LRG maps, saved as mock_lrg_s01_del.hpx2048.fits, etc.
# Must run fetch_WebSky_halos.sh and make_WebSky_LRG_subcatalogs.py first.

from scipy.special import erfc
import numpy as np
import healpy as hp
import copy
from scipy.interpolate import interp1d

def get_catalog(s):
    fname = 'LRG_halos_s'+str(s)+'.pksc'
    catalog=np.fromfile(fname,count=-1,dtype=np.float32)
    Nhalo = int(catalog.shape[0]/4)
    return np.reshape(catalog,(Nhalo,4))

# columns are: redshift, theta, phi, mass
catalog_s1 = get_catalog(1)
catalog_s2 = get_catalog(2)
catalog_s3 = get_catalog(3)
catalog_s4 = get_catalog(4)

# load dn/dz
fname = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.1/main_lrg_pz_dndz_iron_v0.1_dz_0.01.txt'
xxx = np.genfromtxt(fname)
z = (xxx[:,0]+xxx[:,1])/2
dN_dz_s1 = np.array([z,xxx[:,3]]).T
dN_dz_s2 = np.array([z,xxx[:,4]]).T
dN_dz_s3 = np.array([z,xxx[:,5]]).T
dN_dz_s4 = np.array([z,xxx[:,6]]).T

##############################
# Zheng+07 HOD with best-fits from Sandy's paper
Mc = 10**12.89
M1 = 10**14.08
sigma = 0.27
alpha = 1.20
kappa = 0.65
# 0.68 accounts for M_sun -> M_sun/h
Nbar_central = lambda M: erfc(np.log10(Mc/(0.68*M))/np.sqrt(2)/sigma)/2 
Nbar_sat = lambda M: (np.maximum((0.68*M)-kappa*Mc,0.)/M1)**alpha*Nbar_central(M)
##############################

def make_map(catalog,dN_dz,nside=2048,lnM_min=20,lnM_max=80,mask=None,f=1):
    '''
    Creates mock LRG map. 
    '''
    # impose mass cut 
    # does nothing with default lnM_min/lnM_max values
    lnM =  np.log(catalog[:,3])
    I = np.where((lnM>lnM_min) & (lnM<lnM_max))
    z =     catalog[:,0][I]
    theta = catalog[:,1][I]
    phi =   catalog[:,2][I]
    M =     catalog[:,3][I]
    
    # Apply Zheng et al. weighting
    N_central = 1.-np.ceil(np.random.random(size=len(z)) - Nbar_central(M))
    N_sat = np.random.poisson(lam=Nbar_sat(M))
    weights = N_central + N_sat
    
    # randomly downsample (by factor f) to increase shot noise
    I = np.random.choice(range(len(z)), int(f*len(z)), replace=False)
    z =     z[I]
    theta = theta[I]
    phi =   phi[I]
    weights = weights[I]
    
    # reweight to get dN/dz right
    # first "flatten" the dN/dz distribution
    nbar,zedges = np.histogram(z,bins=100)
    zcs = (zedges[:-1]+zedges[1:])/2.
    weights /= interp1d(zcs,nbar,bounds_error=False,fill_value=1e30)(z)
    # and then rescale to the target dN/dz
    weights *= interp1d(dN_dz[:,0],dN_dz[:,1],bounds_error=False,fill_value=0)(z)
    
    # create mock LRG map
    Map = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside, theta, phi)
    np.add.at(Map, pix, weights)
    delta = Map/np.mean(Map)-1.
    lrg=hp.ma(delta) # mock lrg overdensity
    
    if mask is not None:
        lrg.mask=(mask<0.5)
        lrg -= np.mean(lrg)
        
    return lrg,z,weights

# See View_and_make_mocks.ipynb for a comparison of these
# maps to the real data
delta_s1,z1,weights1 = make_map(catalog_s1,dN_dz_s1,f=0.5)
delta_s2,z2,weights2 = make_map(catalog_s2,dN_dz_s2,f=0.55)
delta_s3,z3,weights3 = make_map(catalog_s3,dN_dz_s3,f=0.55)
delta_s4,z4,weights4 = make_map(catalog_s4,dN_dz_s4,f=0.4)

hp.write_map('mock_lrg_s01_del.hpx2048.fits',delta_s1,overwrite=True)
hp.write_map('mock_lrg_s02_del.hpx2048.fits',delta_s2,overwrite=True)
hp.write_map('mock_lrg_s03_del.hpx2048.fits',delta_s3,overwrite=True)
hp.write_map('mock_lrg_s04_del.hpx2048.fits',delta_s4,overwrite=True)
