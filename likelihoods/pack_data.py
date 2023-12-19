# This file constains several helper functions to 
# nicely pack the (Cgg,Ckg) data and covariances given 
# a set of scale cuts.

import numpy as np
from scipy.interpolate import interp1d
import json
import sys

def get_scale_cuts(data, amin, amax, xmin, xmax):
    """
    Returns scale cuts
    """
    n     = len(amin)
    ell   = np.array(data['ell'])
    acuts = [np.where((ell<=amax[i])&(ell>=amin[i]))[0] for i in range(n)]
    xcuts = [np.where((ell<=xmax[i])&(ell>=xmin[i]))[0] for i in range(n)]
    return acuts, xcuts

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

def pack_cl_wl(data, kapName, galNames, amin, amax, xmin, xmax):
    """
    Packages data from .json file and returns
    window functions for cgg,ckg, and the data vector = 
    (stacked Cgg(1), Ckg(1), Cgg(2), Ckg(2) ...)
    """
    nsamp        = len(galNames)
    acuts, xcuts = get_scale_cuts(data, amin, amax, xmin, xmax)
    wla          = []
    wlx          = []
    odata        = np.array([])
    for i,galName in enumerate(galNames):
        acut,xcut = acuts[i],xcuts[i]
        wla_= get_wl(data,galName,galName)[acut,:]
        cgg = get_cl(data,galName,galName)[acut]
        wlx_= get_wl(data,galName,kapName)[xcut,:]
        ckg = get_cl(data,galName,kapName)[xcut]
        wla.append(wla_)
        wlx.append(wlx_)
        odata = np.concatenate((odata,cgg,ckg))
    return wla,wlx,odata

def pack_cov(data, kapName, galNames, amin, amax, xmin, xmax, verbose=False):
    """
    Package the covariance matrix. 
    """
    # Here we first build the "full covariance"
    # and then apply scale cuts
    nell         = len(data['ell'])
    nsamp        = len(galNames)
    acuts, xcuts = get_scale_cuts(data, amin, amax, xmin, xmax)
    cov = np.zeros((2*nsamp*nell,2*nsamp*nell))
    def get_pair(i):
        name2 = galNames[i//2]
        if i%2==0: name1 = name2
        else:      name1 = kapName
        return name1,name2
    for i in range(2*nsamp):
        for j in range(2*nsamp):
            name1,name2 = get_pair(i)
            name3,name4 = get_pair(j)
            cov_ = get_cov(data,name1,name2,name3,name4)
            cov[nell*i:nell*(i+1),nell*j:nell*(j+1)] = cov_ 
    # scale cuts
    I = []
    for i in range(nsamp): I+=list(nell*2*i+acuts[i])+list(nell*(2*i+1)+xcuts[i])
    if verbose: print('Using these idexes for the covariance matrix',I)
    return cov[:,I][I,:]

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
