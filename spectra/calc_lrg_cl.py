#!/usr/bin/env python3
#
import numpy    as np
import healpy   as hp
import pymaster as nmt
import sys



def prepare_maps(nside,isamp=1):
    """Returns the galaxy and kappa maps and masks."""
    # Read the galaxy density and mask.
    pref = "lrg_s{:02d}".format(isamp)
    hext = ".hpx{:04d}.fits".format(nside)
    gals = hp.read_map(pref+'_del'+hext,dtype=None)
    gmsk = hp.read_map(pref+'_msk'+hext,dtype=None)
    # and the CMB convergence and mask.
    kapp = hp.read_map('P18_lens_kap_filt'+hext,dtype=None)
    kmsk = hp.read_map('P18_lens_msk'+hext,dtype=None)
    # and return the results.
    return( (gals,gmsk,kapp,kmsk) )
    #



def make_thy_cl(nside,isamp):
    """Read best-fit theory (incl. noise) and interpolate to full length."""
    nkk = np.loadtxt("P18_lens_nlkk_filt.txt")
    lpt = np.loadtxt("lrg_s{:02d}_mod.txt".format(isamp))
    # Probably need to extrapolate this to higher
    # and lower ell.  Set high ell power to zero.
    ells = np.arange(3*nside)
    cgg  = np.interp(ells,lpt[:,0],lpt[:,1],right=0)
    ckg  = np.interp(ells,lpt[:,0],lpt[:,2],right=0)
    ckk  = np.interp(ells,nkk[:,0],nkk[:,2],right=0)
    return( (cgg,ckg,ckk) )
    #




def make_bins(nside,LperBin):
    # The ell range is larger than we need to avoid aliasing
    # and we drop low ell to avoid spurious power.
    lmin    = 25
    lmax    = 6000
    ells    = np.arange(lmax,dtype='int32')
    weights = np.ones_like(ells)/float(LperBin)
    weights[ells<lmin] = 0 # Remove low ell.
    # Now generate the bandpower indices, here by brute force.
    # A -1 means that ell value is not included in any bandpower.
    bpws = np.zeros_like(ells) - 1
    ibin = 0
    while LperBin*(ibin+1)+lmin<lmax:
        bpws[LperBin*ibin+lmin:LperBin*(ibin+1)+lmin] = ibin
        ibin += 1
    # And tell NaMaster to set it up.
    bins = nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=weights)
    return(bins)
    #



def pseudo_cl(nside=1024,LperBin=75,lmax=1000,isamp=1):
    """Compute the pseudo-Cl and coupling matrix for galaxies & kappa."""
    # Set up a file prefix.
    pref = "lrg_s{:02d}".format(isamp)
    # Load the maps.
    gals,gmsk,kapp,kmsk = prepare_maps(nside,isamp)
    # Work out bins, extending the high-ell range and zeroing low ell.
    bins   = make_bins(nside,LperBin)
    ell    = bins.get_effective_ells()
    bmax   = np.argmax(np.nonzero(ell<lmax)[0]) + 1
    print("Cutting at index ",bmax," with ell[bmax]=",ell[bmax])
    #
    # Note NaMaster takes care of multiplying our maps by masks.
    #
    # We construct the workspace specifically in order to
    # have access to the bandpower windows.
    # First the galaxy auto-spectrum.
    #
    galxy  = nmt.NmtField(gmsk,[gals])
    wsp    = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(galxy,galxy,bins)
    wla    = wsp.get_bandpower_windows()[0,:bmax,0,:] # Just want W^{00}.
    # Now zero out the small values, to avoid strange extrapolations.
    #wla[np.abs(wla)<1e-9*np.max(wla)]=0
    # and write the window function to a file.
    with open(pref+"_wla.txt","w") as fout:
        fout.write("# Galaxy auto-spectrum window function.\n")
        fout.write("# Binned, observed C_bin is given by this matrix\n")
        fout.write("# times theoretical C_ell:\n")
        fout.write("#   C_bin = \sum_{ell=0}^{Nell-1} W_bin,ell C_ell\n")
        fout.write("# Have Nbin={:d} and Nell={:d}.\n".\
                   format(wla.shape[0],wla.shape[1]))
        for i in range(wla.shape[0]):
            outstr = ""
            for j in range(wla.shape[1]):
                outstr += " {:15.8e}".format(wla[i,j])
            fout.write(outstr+"\n")
    # Now compute the galaxy autospectrum.
    gauto  = nmt.compute_full_master(galxy,galxy,bins,workspace=wsp)
    cgg    = gauto[0][:bmax]
    #
    # Now the cross-spectrum.
    #
    kappa  = nmt.NmtField(kmsk,[kapp])
    wsp    = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(galxy,kappa,bins)
    wlx    = wsp.get_bandpower_windows()[0,:bmax,0,:] # Just want W^{00}.
    # Now zero out the small values, to avoid strange extrapolations.
    #wlx[np.abs(wlx)<1e-9*np.max(wlx)]=0
    # and write the window function to a file.
    with open(pref+"_wlx.txt","w") as fout:
        fout.write("# Galaxy-kappa cross-spectrum window function.\n")
        fout.write("# Binned, observed C_bin is given by this matrix\n")
        fout.write("# times theoretical C_ell:\n")
        fout.write("#   C_bin = \sum_{ell=0}^{Nell-1} W_bin,ell C_ell\n")
        fout.write("# Have Nbin={:d} and Nell={:d}.\n".\
                   format(wlx.shape[0],wlx.shape[1]))
        for i in range(wlx.shape[0]):
            outstr = ""
            for j in range(wlx.shape[1]):
                outstr += " {:15.8e}".format(wlx[i,j])
            fout.write(outstr+"\n")
    # Now compute the cross-spectrum.
    cross  = nmt.compute_full_master(galxy,kappa,bins,workspace=wsp)
    ckg    = cross[0][:bmax]
    # Correct the C_l for the pixel window function.  This correction
    # is really only good up to ell~nside, but making it is better than
    # not making it!
    sn,ns  = 5.5e-07,2048   # For LRGs.
    if isamp==0: sn = 6.027057e-07
    if isamp==1: sn = 4.019633e-06
    if isamp==2: sn = 2.242553e-06
    if isamp==3: sn = 2.069106e-06
    if isamp==4: sn = 2.263650e-06
    if isamp==24: sn = 2.181369e-06
    pixwin = hp.pixwin(ns)
    pixwin = np.interp(np.arange(wla.shape[1]),\
                       np.arange(3*ns),pixwin,right=0.5)
    cgg    = sn + (cgg - sn)*np.dot(wla,1/pixwin**2)
    ckg   *= np.dot(wlx,1/pixwin)
    # and write the answers to file.
    with open(pref+"_cls.txt","w") as fout:
        fout.write("# Galaxy auto- and CMB lensing cross-spectra.\n")
        fout.write("# Computed with NaMaster compute_full_master\n")
        fout.write("# at Nside={:d}, last bin {:.1f}\n".format(nside,ell[-1]))
        fout.write("# with pixel window fn correction on C_a-SN and C_x.\n")
        fout.write("# Used SN={:e}.\n".format(sn))
        fout.write("# {:>6s} {:>15s} {:>15s}\n".format("ell","C_l_a","C_l_x"))
        for i in range(cgg.size):
            fout.write("{:8.1f} {:15.5e} {:15.5e}\n".\
                       format(ell[i],cgg[i],ckg[i]))
    #






def gaussian_cov(nside=1024,LperBin=75,lmax=1000,isamp=1):
    """Compute a Gaussian covariance matrix.  This can also be done
       pretty efficiently using Monte-Carlo and synfast."""
    print("Using NaMaster covariance.")
    # Set up a file prefix.
    pref = "lrg_s{:02d}".format(isamp)
    # Load the maps.
    gals,gmsk,kapp,kmsk = prepare_maps(nside,isamp)
    # Set up the fields.
    galxy= nmt.NmtField(gmsk,[gals])
    kappa= nmt.NmtField(kmsk,[kapp])
    # Assign bins and compute workspaces.
    bins = make_bins(nside,LperBin)
    ell  = bins.get_effective_ells()
    bmax = np.argmax(np.nonzero(ell<lmax)[0]) + 1
    print("Cutting at index ",bmax," with ell[bmax]=",ell[bmax])
    # Need theory input spectra of full length.  These
    # spectra contain the stochastic noise terms.
    cgg,ckg,ckk = make_thy_cl(nside,isamp)
    #
    # Start with the auto-auto covariance.
    #
    # Need an Nmt and a covariance workspace:
    wsa= nmt.NmtWorkspace()
    wsa.compute_coupling_matrix(galxy,galxy,bins)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(galxy,galxy,galxy,galxy)
    # Then compute Cov.
    covar_aa = nmt.gaussian_covariance(cw,0,0,0,0,\
                   [cgg],[cgg],[cgg],[cgg],wa=wsa,wb=wsa)
    # We want to increase the error on the first auto-spectrum bin
    # to account for RSD.
    wla = wsa.get_bandpower_windows()[0,0,0,:]
    covar_aa[0,0] += (0.10*np.dot(wla,cgg))**2
    #
    # Next do the cross-cross covariance.
    #
    # Need an Nmt and a covariance workspace:
    wsx= nmt.NmtWorkspace()
    wsx.compute_coupling_matrix(galxy,kappa,bins)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(galxy,kappa,galxy,kappa)
    # Then compute Cov.
    covar_xx = nmt.gaussian_covariance(cw,0,0,0,0,\
                   [cgg],[ckg],[ckg],[ckk],wa=wsx,wb=wsx)
    #
    # Finally do the auto-cross covariance.
    # We have the Nmt workspace already.
    #
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(galxy,galxy,galxy,kappa)
    # Then compute Cov.
    covar_ax = nmt.gaussian_covariance(cw,0,0,0,0,\
                   [cgg],[ckg],[cgg],[ckg],wa=wsa,wb=wsx)
    # Now restrict the bins to impose the lmax cut.
    covar_aa = covar_aa[:bmax,:bmax]
    covar_xx = covar_xx[:bmax,:bmax]
    covar_ax = covar_ax[:bmax,:bmax]
    covar_xa = covar_ax.T
    # Pack Cov.
    Nbin  = covar_aa.shape[0]
    covar = np.zeros( (2*Nbin,2*Nbin) )
    covar[:Nbin,:Nbin] = covar_aa.copy()
    covar[Nbin:,Nbin:] = covar_xx.copy()
    covar[:Nbin,Nbin:] = covar_ax.copy()
    covar[Nbin:,:Nbin] = covar_xa.copy()
    # and write the answers to file.
    with open(pref+"_cov.txt","w") as fout:
        fout.write("# Galaxy-kappa 2x2pt covariance.\n")
        fout.write("# Using NaMaster Gaussian covariance.\n")
        fout.write("# Using {:d} ell bins for both gg and kg.\n".format(Nbin))
        fout.write("# with stacked data vector [Cgg,Ckg].\n")
        for i in range(covar.shape[0]):
            outstr = ""
            for j in range(covar.shape[1]):
                outstr += " {:25.15e}".format(covar[i,j])
            fout.write(outstr+"\n")
    #








def approximate_cov(nside=1024,LperBin=75,lmax=1000,isamp=1):
    """Compute an approximation to the Gaussian covariance matrix.
       This is significantly faster than the in-built NaMaster routine
       while still being reasonably accurate:
       https://arxiv.org/abs/astro-ph/0105302 , S 3."""
    print("Approximating covariance.")
    # Set up a file prefix.
    pref = "lrg_s{:02d}".format(isamp)
    # Load the maps.
    gals,gmsk,kapp,kmsk = prepare_maps(nside,isamp)
    # Set up the fields.
    galxy= nmt.NmtField(gmsk,[gals])
    kappa= nmt.NmtField(kmsk,[kapp])
    # Assign bins and compute workspaces.
    bins = make_bins(nside,LperBin)
    ell  = bins.get_effective_ells()
    bmax = np.argmax(np.nonzero(ell<lmax)[0]) + 1
    print("Cutting at index ",bmax," with ell[bmax]=",ell[bmax])
    # We want the bandpower windows for auto- and cross.
    # This actually takes a while to compute, we should share it.
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(galxy,galxy,bins)
    wla = wsp.get_bandpower_windows()[0,:bmax,0,:] # Just want W^{00}.
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(galxy,kappa,bins)
    wlx = wsp.get_bandpower_windows()[0,:bmax,0,:] # Just want W^{00}.
    # Work out the powers of the window function, w_i.
    # First the galaxies.
    thresh = 0.01   # An arbitrary, small positive number.
    tmsk   = gmsk[gmsk>thresh]
    w2,w4  = np.mean(tmsk**2),np.mean(tmsk**4)
    fsky_g = np.sum(tmsk)/len(gmsk) * w2**2/w4
    print("Galaxy w2={:7.4f}, w4={:7.4f}, fsky={:7.4f}".format(w2,w4,fsky_g))
    # and then kappa.
    tmsk   = kmsk[kmsk>thresh]
    w2,w4  = np.mean(tmsk**2),np.mean(tmsk**4)
    fsky_k = np.sum(tmsk)/len(kmsk) * w2**2/w4
    print("Kappa  w2={:7.4f}, w4={:7.4f}, fsky={:7.4f}".format(w2,w4,fsky_k))
    # Need theory input spectra of full length.  These
    # spectra contain the stochastic noise terms.
    cgg,ckg,ckk = make_thy_cl(nside,isamp)
    #
    # Now we have the Gaussian, diagonal covariances, per ell.
    lfact  = 1.0/(2*np.arange(3*nside)+1)
    var_aa = np.diag(2*cgg**2*lfact/fsky_g)
    var_xx = np.diag((cgg*ckk+ckg**2)*lfact/np.sqrt(fsky_g*fsky_k))
    var_ax = np.diag(2*cgg*ckg*lfact/np.sqrt(fsky_g*fsky_k))
    # bin these using the bandpower weights.
    covar_aa = np.dot(wla,np.dot(var_aa,wla.T))
    covar_xx = np.dot(wlx,np.dot(var_xx,wlx.T))
    covar_ax = np.dot(wla,np.dot(var_ax,wlx.T))
    covar_xa = covar_ax.T
    # Pack the full Cov.
    Nbin  = covar_aa.shape[0]
    covar = np.zeros( (2*Nbin,2*Nbin) )
    covar[:Nbin,:Nbin] = covar_aa.copy()
    covar[Nbin:,Nbin:] = covar_xx.copy()
    covar[:Nbin,Nbin:] = covar_ax.copy()
    covar[Nbin:,:Nbin] = covar_xa.copy()
    # We want to increase the error on the first auto-spectrum bin
    # to account for RSD.
    covar[0,0] += (0.10*np.dot(wla[0,:],cgg))**2
    # and write the answers to file.
    with open(pref+"_cov.txt","w") as fout:
        fout.write("# Galaxy-kappa 2x2pt covariance.\n")
        fout.write("# Using approximate Gaussian covariance.\n")
        fout.write("# Using {:d} ell bins for both gg and kg.\n".format(Nbin))
        fout.write("# with stacked data vector [Cgg,Ckg].\n")
        for i in range(covar.shape[0]):
            outstr = ""
            for j in range(covar.shape[1]):
                outstr += " {:25.15e}".format(covar[i,j])
            fout.write(outstr+"\n")
    #





if __name__=="__main__":
    if len(sys.argv)==1:
        isamp = 1   # Default.
    elif len(sys.argv)==2:
        isamp = int(sys.argv[1])
    else:
        raise RuntimeError("Usage: "+sys.argv[0]+" [isamp]")
    nside,LperBin,lmax = 2048,50,1000
    pseudo_cl(nside,LperBin,lmax,isamp)
    gaussian_cov(nside,LperBin,lmax,isamp)
    #approximate_cov(nside,LperBin,lmax,isamp)
    #
