import numpy as np
import sys
import json
from os.path import exists
import healpy as hp
from healpy.rotator import Rotator
from mpi4py import MPI
from glob import glob

from lensing_sims import get_kappa_maps

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nproc = comm.Get_size()

def measure_cls_anafast(Isim,gal_msk,kap_msk,lensmap,option='baseline',NSIDE_OUT=2048,lmax=2000):
    """
    C_gkt: cross-correlation of "galaxies" (input kappa map masked with gal map)
           and input kappa map (masked with kappa mask)
    C_gkr: cross-correlation of "galaxies" and reconstructed input map 
           (masked with kap mask)
    """
    kap_rec,kap_true = get_kappa_maps(Isim,NSIDE_OUT,lensmap,option=option)
    
    g_proxy  = kap_true * gal_msk; g_proxy  -= np.mean(g_proxy)
    k_true   = kap_true * kap_msk; k_true   -= np.mean(k_true)
    k_recmsk = kap_rec  * kap_msk; k_recmsk -= np.mean(k_recmsk)
    
    C_gkt    = hp.anafast(g_proxy,map2=k_true  ,lmax=lmax,use_pixel_weights=True) 
    C_gkr    = hp.anafast(g_proxy,map2=k_recmsk,lmax=lmax,use_pixel_weights=True)
    
    ell = np.arange(len(C_gkt))
    dat = np.array([ell,C_gkt,C_gkr]).T
    
    return dat
    
def make_mc_cls(gal_name,gal_msk,kap_msk,COORD_IN,NSIDE_OUT=2048,lensmap='PR3',option='baseline'):
    """
    measure cls (for all simulations) and save them to sims/
    """
    if   lensmap == 'PR3':
        rot = Rotator(coord=f'{COORD_IN}g')
        simidx = range(300)
    elif lensmap == 'PR4':
        rot = Rotator(coord=f'{COORD_IN}g')
        simidx = np.array(list(range(60,300)) + list(range(360,600)))
    elif lensmap == 'DR6':
        rot = Rotator(coord=f'{COORD_IN}c')
        simidx = range(1,401)
    else:
        print('ERROR: lensmap must be PR3, PR4 or DR6',flush=True)
        sys.exit()    
    kap_msk = rot.rotate_map_pixel(kap_msk)
    gal_msk = rot.rotate_map_pixel(gal_msk)  
    # run individual sims
    for i in simidx: 
        if i%nproc==rank:
            fname = f'sims/{gal_name}_{lensmap}-{option}_{i}.txt'
            if not exists(fname):
                dat = measure_cls_anafast(i,gal_msk,kap_msk,lensmap,option=option,NSIDE_OUT=NSIDE_OUT)
                np.savetxt(fname,dat,header='Columns are: ell, C_gkt, C_gkr')

def bin_mc_corr(prefix,ledges=[25.+50*i for i in range(21)],lmax=2000):
    """
    compute the MC correction for bandpowers defined by ledges
    """
    # now average
    nbin     = len(ledges)-1
    fnames   = list(glob(f'{prefix}_*'))
    centers  = [(ledges[i]+ledges[i+1])/2 for i in range(nbin)]
    data_gkt = []
    data_gkr = []
    ell,_,_ = np.genfromtxt(fnames[0])[:,:3].T
    for fn in fnames:
        ell,C_gkt,C_gkr = np.genfromtxt(fn)[:,:3].T
        data_gkt.append(C_gkt)
        data_gkr.append(C_gkr)
    Ckgt = np.mean(data_gkt,axis=0)
    Ckgr = np.mean(data_gkr,axis=0)
    Ckgt_bin = np.ones(nbin)
    Ckgr_bin = np.ones(nbin)
    for i in range(nbin):
        I = np.where((ell>=ledges[i]) & (ell<ledges[i+1]))
        if len(I[0])>0 and (ledges[i+1]<lmax):
            Ckgt_bin[i] = np.mean(Ckgt[I])
            Ckgr_bin[i] = np.mean(Ckgr[I])
    dat = np.array([centers,Ckgt_bin/Ckgr_bin]).T
    return dat

def apply_mc_corr(fnin,fnout,kapName,galNames,mccorr_prefixs):
    """
    multiply cl's in json file by MC norm correction
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