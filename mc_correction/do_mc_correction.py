import numpy as np
import sys
from os.path import exists
import healpy as hp
from healpy.rotator import Rotator
from mpi4py import MPI
from lensing_sims import get_kappa_maps
from glob import glob

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nproc = comm.Get_size()


def measure_cls_anafast(Isim,gal_msk,kap_msk,lensmap,NSIDE_OUT=2048,lmax=2000):
    """
    """
    kap_recon,kap_true = get_kappa_maps(Isim,NSIDE_OUT,lensmap)
    
    g_proxy = kap_true  * gal_msk; g_proxy -= np.mean(g_proxy)
    k_true  = kap_true  * kap_msk; k_true  -= np.mean(k_true)
    k_recon = kap_recon * kap_msk; k_recon -= np.mean(k_recon)
    
    C_gkt = hp.anafast(g_proxy,map2=k_true ,lmax=lmax,use_pixel_weights=True) 
    C_gkr = hp.anafast(g_proxy,map2=k_recon,lmax=lmax,use_pixel_weights=True)
    ell   = np.arange(len(C_gkt))
    
    dat   = np.array([ell,C_gkt,C_gkr]).T
    
    return dat
    
    
def do_mc_correction(gal_name,gal_msk,kap_msk,COORD_IN,NSIDE_OUT=2048,lensmap='PR3',ledges=[25.+50*i for i in range(nbin+1)]):
    """
    assumes that COORD_IN is the same coordinate system as the CMB lensing 
    maps in ../maps/masks/
    """
    if lensmap == 'PR3':
        rot = Rotator(coord=f'{COORD_IN}g')
        simidx = range(300)
    elif lensmap == 'PR4':
        rot = Rotator(coord=f'{COORD_IN}g')
        simidx = np.array(list(range(60,300)) + list(range(360,600)))
    elif lensmap == 'ACT':
        rot = None
        simidx = None
    else:
        print('ERROR: lensmap must be PR3, PR4 or ACT',flush=True)
        sys.exit()    
    kap_msk = rot.rotate_map_pixel(kap_msk)
    gal_msk = rot.rotate_map_pixel(gal_msk)  
    # run individual sims
    for i in simidx: 
        if i%nproc==rank:
            fname = f'sims/{gal_name}_{lensmap}_{i}.txt'
            if not exists(fname):
                dat = measure_cls_anafast(i,gal_msk,kap_msk,lensmap,NSIDE_OUT=NSIDE_OUT)
                np.savetxt(fname,dat,header='Columns are: ell, C_gkt, Cgkr')
    # now average
    fnames = list(glob(f'sims/{gal_name}_{lensmap}_*'))
    centers = [(ledges[i]+ledges[i+1])/2 for i in range(nbin)]
    data_gkt = []
    data_gkr = []
    for fn in fnames:
        ell,C_gkt,C_gkr = np.genfromtxt(fn).T
        data_gkt.append(C_gkt)
        data_gkr.append(C_gkr)
    Ckgt = np.mean(data_gkt,axis=0)
    Ckgr = np.mean(data_gkr,axis=0)
    Ckgt_bin = np.zeros(nbin)
    Ckgr_bin = np.zeros(nbin)
    for i in range(nbin):
        I = np.where((ell>=ledges[i]) & (ell<ledges[i+1]))
        Ckgt_bin[i] = np.mean(Ckgt[I])
        Ckgr_bin[i] = np.mean(Ckgr[I])
    dat = np.array([centers,Ckgt_bin/Ckgr_bin]).T
    np.savetxt(f'MC_correction_{gal_name}_{lensmap}.txt',dat,header=f'Results for {len(fnames)} {lensmap} sims. Columns are: ell, Ckgt/Ckgr')
 

if __name__ == "__main__":
    isamp    = 1
    lensmap  = 'PR3'
    bdir     = '/pscratch/sd/m/mwhite/DESI/MaPar/maps/'
    lrg_mask = hp.read_map(bdir+f'lrg_s0{isamp}_msk.hpx2048.fits')
    kap_mask = hp.read_map(f'../maps/masks/{lensmap}_lens_mask.fits',dtype=None)
    lrg_name = f'LRG_full_z{isamp}'
    
    do_mc_correction(lrg_name,lrg_mask,kap_mask,'c',lensmap=lensmap)
    