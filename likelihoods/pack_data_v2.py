# This file constains several helper functions to 
# nicely pack the (Cgg,Ckg) data and covariances given 
# a set of scale cuts.

import numpy as np
from scipy.interpolate import interp1d
import json
import sys

def pack_dndz(dndzs,zmin=0.05,nz=500):
    """
    Package the redshift distributions.
    Cuts off the redshift distribution below z<zmin
    """
    n  = len(dndzs)
    zs = [dndzs[i][:,0] for i in range(n)]
    zmax = np.max(np.concatenate(zs).flatten())
    zeval = np.linspace(zmin,zmax,nz)
    dndz = np.zeros((len(zeval),n+1))
    dndz[:,0] = zeval
    for i in range(n): dndz[:,i+1] = interp1d(dndzs[i][:,0],dndzs[i][:,1],bounds_error=False,fill_value=0.)(zeval)
    return dndz

def get_cl(data,name1,name2):
    """
    Returns the measured power spectrum.
    (agnostic towards the order of the names in data)
    """
    try:    return np.array(data[f'cl_{name1}_{name2}'])
    except: return np.array(data[f'cl_{name2}_{name1}'])

def get_wl(data,name1,name2):
    """
    Returns the measured window function.
    (agnostic towards the order of the names in data)
    """
    try:    return np.array(data[f'wl_{name1}_{name2}'])
    except: return np.array(data[f'wl_{name2}_{name1}'])

def get_cov(data,name1,name2,name3,name4):
    """
    Returns the covariance of C_{name1}_{name2} with C_{name3}_{name4}. 
    (agnostic towards the order of the names in data)
    """
    def tryit(pair1,pair2,transpose=False):
        try:
            res = np.array(data[f'cov_{pair1}_{pair2}'])
            if transpose: res = res.T
            return res
        except:
            return -1
    perms12 = [f'{name1}_{name2}',f'{name2}_{name1}']
    perms34 = [f'{name3}_{name4}',f'{name4}_{name3}']
    for i in range(2):
        for j in range(2):
            res = tryit(perms12[i],perms34[j])
            if not isinstance(res,int): return res
            res = tryit(perms34[i],perms12[j],transpose=True)
            if not isinstance(res,int): return res
    print(f'Error: cov_{perms12[0]}_{perms34[0]}, or any equivalent permutation')
    print( 'of the names, is not found in the data')
    sys.exit()
    
def get_scale_cuts(data, amin, amax, xmin, xmax):
    """
    Returns scale cuts
    """
    n     = len(amin)
    m     = len(xmin)
    ell   = np.array(data['ell'])
    acuts = [np.where((ell<=amax[i])&(ell>=amin[i]))[0] for i in range(n)]
    xcuts = [[np.where((ell<=xmax[j][i])&(ell>=xmin[j][i]))[0] for i in range(n)] for j in range(m)]
    return acuts, xcuts
    
def pack_cl_wl(data, kapNames, galNames, amin, amax, xmin, xmax):
    """
    Packages data from .json file and returns
    window functions and the data vector.
    
    If kapNames = [k1,k2,...,kn] and galNames = [g1,g2,...,gm] then the 
    data vector is concatenate(Cg1g1,Ck1g1,...,Ckng1,Cg2g2,Ck1g2,...,Ckngm)
    """
    nsamp        = len(galNames)
    acuts, xcuts = get_scale_cuts(data, amin, amax, xmin, xmax)
    wla          = []
    wlx          = []
    odata        = np.array([])
    # cl,wla
    for i,galName in enumerate(galNames):
        wla.append(get_wl(data,galName,galName)[acuts[i],:])
        cgg   = get_cl(data,galName,galName)[acuts[i]]
        odata = np.concatenate((odata,cgg))
        for j,kapName in enumerate(kapNames):
            ckg   = get_cl(data,galName,kapName)[xcuts[j][i]]
            odata = np.concatenate((odata,ckg))
    # wlx 
    for j,kapName in enumerate(kapNames):
        wlx_ = []
        for i,galName in enumerate(galNames): 
            wlx_.append(get_wl(data,kapName,galName)[xcuts[j][i],:])
        wlx.append(wlx_)  
    
    return wla,wlx,odata

def pack_cov(data, kapNames, galNames, amin, amax, xmin, xmax, verbose=False):
    """
    Package the covariance matrix.
    
    If kapNames = [k1,k2,...,kn] and galNames = [g1,g2,...,gm] then the basis
    for the covariance is (Cg1g1,Ck1g1,...,Ckng1,Cg2g2,Ck1g2,...,Ckngm)
    """
    # Build the full covariance (all ell's)
    nell         = len(data['ell'])
    nsamp        = len(galNames)
    nkap         = len(kapNames)
    acuts, xcuts = get_scale_cuts(data, amin, amax, xmin, xmax)
    cov = np.zeros(((1+nkap)*nsamp*nell,(1+nkap)*nsamp*nell))
    def get_pair(i):
        name2 = galNames[i//(1+nkap)]
        if i%(1+nkap) == 0: name1 = name2
        else: name1 = kapNames[i%(1+nkap) - 1]
        return name1,name2
    for i in range((1+nkap)*nsamp):
        for j in range((1+nkap)*nsamp):
            name1,name2 = get_pair(i)
            name3,name4 = get_pair(j)
            cov_ = get_cov(data,name1,name2,name3,name4)
            cov[nell*i:nell*(i+1),nell*j:nell*(j+1)] = cov_ 
    # and then apply scale cuts
    I = []
    for i in range(nsamp):
        I += list(nell*(1+nkap)*i+acuts[i])
        for j in range(nkap):
           I += list(nell*(1+nkap)*i+nell*(1+j)+xcuts[j][i])
    if verbose: print('Using these idexes for the covariance matrix',I)
    return cov[:,I][I,:]
