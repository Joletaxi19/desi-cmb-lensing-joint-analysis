#!/usr/bin/env python3
# makes the PR4 kappa map and mask in equatorial coordinates
# after low pass filtering
# also saves a file to ../data/PR4_lens_nlkk_filt.txt with 
# the low pass filtered noise curves

import numpy    as np
import healpy   as hp
import pymaster as nmt
#
lowpass = True
apodize = True
Nside   = 2048
# Read the data, mask and noise properties.
pl_klm  = np.nan_to_num(hp.read_alm('/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_klm_dat_p.fits'))
pl_mask = hp.ud_grade(hp.read_map('/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/mask.fits.gz',dtype=None),Nside)
pl_nkk  = np.loadtxt( '../data/PR4_lens_nlkk.txt')
#
if lowpass:
    # Filter the alm to remove high ell power.
    lmax   = 2500.
    lval   = np.arange(3*2048)
    xval   = (lval/lmax)**6
    filt   = np.exp(-xval)
    print("Low-pass filtering kappa.")
    print("  : Filter at ell=600  is ",np.interp(600.,lval,filt))
    print("  : Filter at ell=1000 is ",np.interp(1e3,lval,filt))
    print("  : Filter at ell=4000 is ",np.interp(4e3,lval,filt))
    pl_klm = hp.almxfl(pl_klm,filt)
    # Modify the noise curve also -- by the square.
    pl_nkk[:,1] *= np.interp(pl_nkk[:,0],lval,filt**2)
    pl_nkk[:,2] *= np.interp(pl_nkk[:,0],lval,filt**2)
    # and write the modified noise file.
    with open("../data/PR4_lens_nlkk_filt.txt","w") as fout:
        fout.write("# Planck lensing noise curves.\n")
        fout.write("# These curves have been low-pass filtered.\n")
        fout.write("# {:>6s} {:>15s} {:>15s}\n".\
                   format("ell","Noise","Sig+Noise"))
        for i in range(pl_nkk.shape[0]):
            fout.write("{:8.0f} {:15.5e} {:15.5e}\n".\
                       format(pl_nkk[i,0],pl_nkk[i,1],pl_nkk[i,2]))
# rotate from galactic to celestial coordinates
pl_kappa = hp.alm2map(pl_klm,Nside)
rot      = hp.rotator.Rotator(coord='gc')
pl_kappa = rot.rotate_map_pixel(pl_kappa)
pl_mask  = rot.rotate_map_pixel(pl_mask)
#
if apodize: # Apodsize the mask.
    apos    = 0.5  # deg.
    print("Apodizing the mask by {:.2f} deg.".format(apos))
    #pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="Smooth")
    #pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="C1")
    pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="C2")
# Now write the processed maps.
if lowpass:
    outfn= 'PR4_lens_kap_filt.hpx{:04d}.fits'.format(Nside)
else:
    outfn= 'PR4_lens_kap.hpx{:04d}.fits'.format(Nside)
hp.write_map(outfn,pl_kappa,dtype='f4',coord='C',overwrite=True)
outfn    = 'masks/PR4_lens_msk.hpx{:04d}.fits'.format(Nside)
hp.write_map(outfn,hp.ud_grade(pl_mask,Nside),dtype='f4',\
             coord='C',overwrite=True)