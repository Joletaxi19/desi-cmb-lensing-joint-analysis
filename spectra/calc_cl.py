# A wrapper around NaMaster
# Neatly stores all spectra and covariances in a .json format

import numpy    as np
import healpy   as hp
import pymaster as nmt
import sys
import json
from os.path import exists

readme = "cl_X_Y is the pseudo Cell C^{XY}. wl_X_Y is the window function for C^{XY}. "
readme+= "cij is a (nmaps,nmaps,nell) ndarray of spectra used to compute the covariance. "
readme+= "The order of 'map names' defines the order of cij. cov_X_Y_M_N is the covariance of C^{XY} and C^{MN}. "


def get_bins(ledges,nside):
    """
    Takes an input set of ledges and sets up a NaMaster bin object.
    
    ledges : list of ell-bin edges
    nside  : healpix nside
    """
    # set up ell-bins
    Nbin = len(ledges)-1
    ells = np.arange(ledges[-1],dtype='int32')
    bpws = np.zeros_like(ells) - 1
    for i in range(Nbin): bpws[ledges[i]:ledges[i+1]] = i
    bins = nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=np.ones_like(ells))
    return bins

    
def master_cl(ledges,map1,msk1,map2,msk2):
    """
    A bare-bones pseudo-cell calculator. ledges (list)
    defines the edges of the ell-bins. The maps and masks
    are assumed to be in healpix format with the same nside.
    
    Returns the effective-ells, the window function, and the
    pseudo-cells.
    """
    nside = int((len(map1)/12)**0.5)
    bins  = get_bins(ledges,nside)
    field1 = nmt.NmtField(msk1,[map1])
    field2 = nmt.NmtField(msk2,[map2])
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(field1,field2,bins)
    ell = bins.get_effective_ells()
    w12 = wsp.get_bandpower_windows()[0,:,0,:]
    c12 = nmt.compute_full_master(field1,field2,bins,workspace=wsp)[0,:]
    return ell,w12,c12


def master_cov(ledges,map1,msk1,map2,msk2,map3,msk3,map4,msk4,c13,c14,c23,c24):
    """
    A bare-bones covariance calculator. ledges (list)
    defines the edges of the ell-bins. The maps and masks
    are assumed to be in healpix format with the same nside.
    """
    nside = int((len(map1)/12)**0.5)
    bins  = get_bins(ledges,nside)
    field1 = nmt.NmtField(msk1,[map1])
    field2 = nmt.NmtField(msk2,[map2])
    field3 = nmt.NmtField(msk3,[map3])
    field4 = nmt.NmtField(msk4,[map4])
    wsp12 = nmt.NmtWorkspace()
    wsp12.compute_coupling_matrix(field1,field2,bins)
    wsp34 = nmt.NmtWorkspace()
    wsp34.compute_coupling_matrix(field3,field4,bins)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(field1,field2,field3,field4)
    cov = nmt.gaussian_covariance(cw,0,0,0,0,[c13],[c14],[c23],[c24],wa=wsp12,wb=wsp34)  
    return cov


def cij_poly_approx(data):
    """
    Given a set of data (pseudo-cells in a .json file following
    the format of full_master) returns a (nmap,nmap,3*nside)
    ndarray approximating the measured pseudo-cells with 
    an 8-th order polynomial. Extrapolates with zeros for ell
    larger than where the spectra have been measured. 
    
    WARNING: If the .json file is missing a spectrum (e.g. when
    full_master only computes the auto-correlations) then 
    cij_poly_approx "approximates" the spectrum with zeros.
    """
    names  = data['map names']
    nside  = data['nside']
    ell    = np.array(data['ell'])
    nmaps  = len(names)
    result = np.zeros((nmaps,nmaps,3*nside))
    for i in range(nmaps):
        for j in range(i,nmaps):
            try:
                lval  = np.arange(max(ell))
                cl    = np.array(data[f'cl_{names[i]}_{names[j]}'])
                coeff = np.polyfit(ell,cl,8,w=(2*ell+1)**0.5/cl)
                damp  = np.exp(-(10./(lval+1e-6))**6)
                result[i,j,:len(lval)] = np.poly1d(coeff)(lval)*damp
            except:
                continue
    return result
    
    
def full_master(ledges,maps,msks,names,fnout,do_cov=False,cij=None,only_auto=False,overwrite=False):
    """
    Computes power spectra and covariances and saves them in a .json format.
    It is assumed that all maps and masks have the same nside.
    
    ledges   : list of ell-bin edges 
    maps     : list of maps in healpix format
    msks     : list of masks in healpix format
    names    : list of strings to identify each map+mask pair with
    fnout    : string, filename of the output, should end with .json
    do_cov   : to compute the covariance or to not?
    cij      : optional (nmaps,nmaps,3*nside) ndarray. The spectra
               used when estimating the covariance matrix. Order of 
               spectra should match names, i.e. cij[1,2] corresponds to 
               Cell[names[1],names[2]]. Only the upper triangle is used. 
               Approximates spectra with a polynomial fit to the measured 
               spectra if not provided. 
    only_auto: if True, only compute the auto-correlation's of the input maps. 
               If cij isn't provided, assumes that the cross-correlations are
               zero when computing the covariance.
    overwrite: if False (default), doesn't recompute prexisting pseudo-Cells and 
               covariances of the existing .json file
    """
    # enumerate list of unique pairs (upper triangle)
    nmaps = len(maps)
    pairs = []
    for i in range(nmaps):
        for j in range(i,nmaps):
            pairs.append([i,j])
    if only_auto: pairs = [[i,i] for i in range(nmaps)]
    nspec = len(pairs) 
    # and a shortcut to save json file
    def write_outdata(outdata):
        with open(fnout, "w") as outfile:
            json.dump(outdata, outfile, indent=2)
    # infer nside, get effective ell's and 
    # pixel window function
    nside   = int((len(maps[0])/12)**0.5)
    bins    = get_bins(ledges,nside)
    ell     = bins.get_effective_ells()
    pixwin  = hp.pixwin(nside)
    # load or create outdata
    if exists(fnout) and (not overwrite):
        with open(fnout) as outfile:
            outdata = json.load(outfile)
            # update the input names so that it's possible to 
            # add maps to an existing .json file post facto. 
            # However, nside and ledges cannot be updated without
            # overwriting the entire file.
            outdata['map names'] = names
    else:
        outdata = {'README':readme,'nside':nside,'map names':names,
                   'ledges':ledges,'ell':ell.tolist(),'pixwin':pixwin}
    write_outdata(outdata)
    
    # define fields and workspaces
    # compute window functions and pseudo-Cells (if not already computed)
    fields = [nmt.NmtField(msks[i],[maps[i]]) for i in range(nmaps)]
    wsps   = [nmt.NmtWorkspace() for i in range(nspec)]
    for a in range(nspec):
        i,j = pairs[a]
        wsps[a].compute_coupling_matrix(fields[i],fields[j],bins)
        wl_fname = f'wl_{names[i]}_{names[j]}'
        cl_fname = f'cl_{names[i]}_{names[j]}'
        if (cl_fname in outdata) and (not overwrite):
            continue
        else:
            wl = wsps[a].get_bandpower_windows()[0,:,0,:]
            cl = nmt.compute_full_master(fields[i],fields[j],bins,workspace=wsps[a])[0,:]
            outdata[wl_fname] = wl.tolist()
            outdata[cl_fname] = cl.tolist()
            write_outdata(outdata)   

    # start working on the covariance (if applicable)
    if not do_cov: return 'all done'
    if cij is None: cij = cij_poly_approx(outdata)
    outdata['cij'] = cij.tolist()
    write_outdata(outdata)
    
    # compute gaussian covariances
    for a in range(nspec):
        for b in range(a,nspec):
            i,j = pairs[a]
            k,l = pairs[b]
            cov_fname = f'cov_{names[i]}_{names[j]}_{names[k]}_{names[l]}'
            if (cov_fname in outdata) and (not overwrite):
                continue
            else:
                cw = nmt.NmtCovarianceWorkspace()
                cw.compute_coupling_coefficients(fields[i],fields[j],fields[k],fields[l])
                cov = nmt.gaussian_covariance(cw,0,0,0,0,[cij[i,k]],[cij[i,l]],\
                                             [cij[j,k]],[cij[j,l]],wa=wsps[a],wb=wsps[b])  
                outdata[cov_fname] = cov.tolist()
                write_outdata(outdata)
