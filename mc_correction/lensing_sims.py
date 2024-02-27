# Some bookkeeping of lensing reconstruction simulations
# get_kappa_maps is a "catch all" for PR3/PR4/DR6

# PR3 input maps were downloaded from the PLA, should remove my local directory
# and make a script to download these inputs. PR3 reconstruction maps have 
# moved as of Feb 23 2024 (ffp10/lensing no longer exists)

# Planck PR3/PR4 simulations are in galactic coords
# ACT DR6 simulations are in celestial coords

import numpy as np
import healpy as hp

def get_PR3_maps(simidx,nside):
    '''
    Returns reconstructed and true kappa map 
    from a simulation indexed by simidx:
    simidx = 0,...,299
    '''
    # get reconstructed map
    bdir = 'COM_Lensing-SimMap_4096_R3.00/MV/'
    fname = bdir + 'sim_klm_%03d.fits'%simidx
    kappa_sim_alm = np.nan_to_num(hp.read_alm(fname))
    kap_recon = hp.alm2map(kappa_sim_alm,nside)
    
    # get true map
    bdir = '/pscratch/sd/n/nsailer/mc_mult_corr/PR3_lensing_inputs/'
    fname = bdir + 'sky_klm_%03d.fits'%simidx
    true_map_alm = np.nan_to_num(hp.read_alm(fname))
    kap_true = hp.alm2map(true_map_alm,nside)
    
    return kap_recon,kap_true

def get_PR4_maps(simidx,nside):
    '''
    Returns reconstructed and true kappa map 
    from a simulation indexed by simidx:
    simidx = 60,...,300,360,...600
    (not sure why 301,...,359 don't exist)
    '''
    bdir = '/global/cfs/cdirs/cmb/data/'
    bdir_recon = 'planck2020/PR4_lensing/PR4_sims/'
    bdir_truth = 'generic/cmb/ffp10/mc/scalar/'
    
    # get reconstructed map
    fname = bdir+bdir_recon+'klm_sim_%04d_p.fits'%simidx
    kappa_sim_alm = np.nan_to_num(hp.read_alm(fname)) 
    kap_recon = hp.alm2map(kappa_sim_alm,nside)
    
    # get true map 
    fname = bdir+bdir_truth+'ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'%(simidx + 200)
    true_map_alm,mmax = hp.read_alm(fname,hdu=4,return_mmax=True)
    pixel_idx = np.arange(len(true_map_alm))
    L = hp.sphtfunc.Alm.getlm(mmax,i=pixel_idx)[0]
    true_map_alm *= L*(L+1)/2 # phi -> kappa
    kap_true = hp.alm2map(true_map_alm,nside)
    
    return kap_recon,kap_true

def get_DR6_maps(simidx,nside,option='baseline'):
    """
    Returns reconstructed and true kappa map 
    from a simulation indexed by simidx:
    simidx = 1,..,400
    
    option = baseline, cibdeproj, f090, f090_tonly, f150, f150_tonly, 
             galcut040, galcut040_polonly, polonly, tonly
             
    There are other options (like 150 - 90) but I don't care about them for now.
    """
    release = 'dr6_lensing_v1'
    bdir    =f'/global/cfs/projectdirs/act/www/{release}/'
    
    # get reconstructed map
    kappa_rec_alm = np.nan_to_num(hp.read_alm(f'{bdir}maps/{option}/simulations/kappa_alm_sim_act_{release}_{option}_{simidx:04d}.fits'))
    filt          = np.ones(3*nside) ; filt[3000:] = 0. 
    kappa_rec_alm = hp.almxfl(kappa_rec_alm,filt)
    kap_recon     = hp.alm2map(kappa_rec_alm,nside)
    
    # get true map
    true_map_alm  = np.nan_to_num(hp.read_alm(f"{bdir}sim_inputs/kappa_alm/input_kappa_alm_sim_{simidx:04d}.fits"))
    true_map_alm  = hp.almxfl(true_map_alm,filt)
    kap_true      = hp.alm2map(true_map_alm,nside)

    return kap_recon,kap_true

def get_kappa_maps(simidx,nside,lensmap,option='baseline'):
    if lensmap == 'PR3'  : return get_PR3_maps(simidx,nside)
    if lensmap == 'PR4'  : return get_PR4_maps(simidx,nside)
    if lensmap == 'DR6'  : return get_DR6_maps(simidx,nside,option=option)
    print('ERROR: lensmap must be PR3, PR4 or DR6',flush=True)