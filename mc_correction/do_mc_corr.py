import numpy as np
import sys
import json
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
    
    
def make_mc_cls(gal_name,gal_msk,kap_msk,COORD_IN,NSIDE_OUT=2048,lensmap='PR3'):
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


def bin_mc_corr(prefix,ledges=[25.+50*i for i in range(21)]):
    # now average
    nbin = len(ledges)-1
    fnames = list(glob(f'{prefix}_*'))
    centers = [(ledges[i]+ledges[i+1])/2 for i in range(nbin)]
    data_gkt = []
    data_gkr = []
    ell,_,_ = np.genfromtxt(fnames[0]).T
    for fn in fnames:
        ell,C_gkt,C_gkr = np.genfromtxt(fn).T
        data_gkt.append(C_gkt)
        data_gkr.append(C_gkr)
    Ckgt = np.mean(data_gkt,axis=0)
    Ckgr = np.mean(data_gkr,axis=0)
    Ckgt_bin = np.ones(nbin)
    Ckgr_bin = np.ones(nbin)
    for i in range(nbin):
        I = np.where((ell>=ledges[i]) & (ell<ledges[i+1]))
        if len(I[0])>0:
            Ckgt_bin[i] = np.mean(Ckgt[I])
            Ckgr_bin[i] = np.mean(Ckgr[I])
    dat = np.array([centers,Ckgt_bin/Ckgr_bin]).T
    return dat


def apply_mc_corr(fnin,fnout,kapName,galNames,mccorr_prefixs):
    """
    """
    with open(fnin) as indata:
        data = json.load(indata)
    for i,galName in enumerate(galNames):
        mccorr = bin_mc_corr(mccorr_prefixs[i],data['ledges'])[:,1]
        data[f'mccorr_{kapName}_{galName}'] = mccorr.tolist()
        try:
            name = f'cl_{kapName}_{galName}'
            data[name] = (np.array(data[name])*mccorr).tolist()
        except:
            name = f'cl_{galName}_{kapName}'
            data[name] = (np.array(data[name])*mccorr).tolist()
    with open(fnout, "w") as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    isamp   = 1
    lensmap  = 'PR3'
    bdir     = '/pscratch/sd/m/mwhite/DESI/MaPar/maps/'
    lrg_mask = hp.read_map(bdir+f'lrg_s0{isamp}_msk.hpx2048.fits')
    kap_mask = hp.read_map(f'../maps/masks/{lensmap}_lens_mask.fits',dtype=None)
    north    = hp.read_map('../maps/masks/north_mask.fits')
    des_mask = hp.read_map('../maps/masks/des_mask.fits')
    dec15    = hp.read_map('../maps/masks/DECm15_mask.fits')
    
    # LRG full footprint x PR3
    if True:
        lrg_name = f'LRG_full_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG full x PR3',flush=True)    
    # LRG North footprint x PR3
    if True:
        lrg_name = f'LRG_north_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask*north,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG north x PR3',flush=True)
    # LRG DES plus footprint x PR3
    if True:
        lrg_name = f'LRG_desp_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask*des_mask*(1-dec15),kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG DES plus x PR3',flush=True)
    # LRG DES minus footprint x PR3
    if True:
        lrg_name = f'LRG_desm_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask*des_mask*dec15,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG DES minus x PR3',flush=True)
    # LRG DECaLS footprint x PR3
    if True:
        lrg_name = f'LRG_decals_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask*decals,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG DECaLS x PR3',flush=True)
    # LRG DES footprint x PR3
    if True:
        lrg_name = f'LRG_des_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask*des_mask,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG DES x PR3',flush=True)

    lensmap  = 'PR4'
    kap_mask = hp.read_map(f'../maps/masks/{lensmap}_lens_mask.fits',dtype=None)
    
    # LRG full footprint x PR4
    if True:
        lrg_name = f'LRG_full_z{isamp}'    
        make_mc_cls(lrg_name,lrg_mask,kap_mask,'c',lensmap=lensmap)
        print('Done with MC sims for LRG full x PR4',flush=True)  