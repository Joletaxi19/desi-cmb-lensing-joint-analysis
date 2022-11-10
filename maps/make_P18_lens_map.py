#!/usr/bin/env python3
#
# Prepare the Planck (2018) lensing maps from the files provided at
# https://pla.esac.esa.int/ cosmology products, lensing products
# as COM_Lensing_4096_R3.00.tgz
#
#
import numpy    as np
import healpy   as hp
import pymaster as nmt
#
lowpass = True
apodize = True
# Read the data, mask and noise properties.
pl_klm  = hp.read_alm('P18_lens_klm.fits')
pl_mask = hp.read_map('P18_lens_msk.fits',dtype=None)
pl_nkk  = np.loadtxt( 'P18_lens_nlkk.txt')
#
if lowpass:
    # Filter the alm to remove high ell power.
    lmax   = 2500.
    lval   = np.arange(3*2048)
    xval   = (lval/lmax)**6
    filt   = np.exp(-xval)
    print("Low-pass filtering kappa.")
    print("  : Filter at ell=1000 is ",np.interp(1e3,lval,filt))
    print("  : Filter at ell=4000 is ",np.interp(4e3,lval,filt))
    pl_klm = hp.almxfl(pl_klm,filt)
    # Modify the noise curve also -- by the square.
    pl_nkk[:,1] *= np.interp(pl_nkk[:,0],lval,filt**2)
    pl_nkk[:,2] *= np.interp(pl_nkk[:,0],lval,filt**2)
    # and write the modified noise file.
    with open("P18_lens_nlkk_filt.txt","w") as fout:
        fout.write("# Planck lensing noise curves.\n")
        fout.write("# These curves have been low-pass filtered.\n")
        fout.write("# {:>6s} {:>15s} {:>15s}\n".\
                   format("ell","Noise","Sig+Noise"))
        for i in range(pl_nkk.shape[0]):
            fout.write("{:8.0f} {:15.5e} {:15.5e}\n".\
                       format(pl_nkk[i,0],pl_nkk[i,1],pl_nkk[i,2]))
#
if apodize: # Apodsize the mask.
    apos    = 0.5  # deg.
    print("Apodizing the mask by {:.2f} deg.".format(apos))
    #pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="Smooth")
    #pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="C1")
    pl_mask = nmt.mask_apodization(pl_mask,apos,apotype="C2")
# Now write the processed maps at different Nside.
for Nside in [1024,2048]:
    pl_kappa = hp.alm2map(pl_klm,Nside)
    if lowpass:
        outfn= 'P18_lens_kap_filt.hpx{:04d}.fits'.format(Nside)
    else:
        outfn= 'P18_lens_kap.hpx{:04d}.fits'.format(Nside)
    hp.write_map(outfn,pl_kappa,dtype='f4',coord='G',overwrite=True)
    outfn    = 'P18_lens_msk.hpx{:04d}.fits'.format(Nside)
    hp.write_map(outfn,hp.ud_grade(pl_mask,Nside),dtype='f4',\
                 coord='G',overwrite=True)
#
