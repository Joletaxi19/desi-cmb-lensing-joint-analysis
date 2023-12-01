import numpy    as np
import healpy   as hp
import pymaster as nmt
import os
import urllib.request
import sys
sys.path.append('../')
from globe import NSIDE,COORD
#
lowpass = True
Nside   = NSIDE
# Read the data, mask and noise properties.
bdir    = '/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/'
pl_klm  = np.nan_to_num(hp.read_alm(bdir+'PR4_klm_dat_p.fits'))
pl_mask = hp.ud_grade(hp.read_map(bdir+'mask.fits.gz',dtype=None),Nside)
nkk     = np.nan_to_num(np.loadtxt(bdir+'PR4_nlkk_p.dat'))
# download data 
# Read the data, mask and noise properties.
# download PR3 data to get fiducial ckk
website = 'http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID='
fname   = 'COM_Lensing_4096_R3.00' #COM_Lensing-Szdeproj_4096_R3.00
urllib.request.urlretrieve(website+fname+'.tgz', fname+'.tgz')
os.system(f"tar -xvzf {fname}.tgz")  
os.remove(fname+'.tgz')
pr3_nkk  = np.loadtxt(f'{fname}/MV/nlkk.dat')
os.system(f"rm -r {fname}")
# 
pl_nkk      = np.zeros((len(nkk),3))
pl_nkk[:,0] = np.arange(len(nkk))
pl_nkk[:,1] = nkk
pl_nkk[:,2] = (pr3_nkk[:,2]-pr3_nkk[:,1])[:len(nkk)] + nkk
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
rot      = hp.rotator.Rotator(coord=f'g{COORD}')
pl_kappa = rot.rotate_map_pixel(pl_kappa)
pl_mask  = rot.rotate_map_pixel(pl_mask)
#
apos    = 0.5  # deg.
print("Apodizing the mask by {:.2f} deg.".format(apos))
pl_mask_apod = nmt.mask_apodization(pl_mask,apos,apotype="C2")
# Now write the processed maps.
if lowpass:
    outfn= 'PR4_lens_kap_filt.hpx{:04d}.fits'.format(Nside)
else:
    outfn= 'PR4_lens_kap.hpx{:04d}.fits'.format(Nside)
hp.write_map(outfn,pl_kappa,dtype='f4',coord='C',overwrite=True)
outfn    = 'masks/PR4_lens_mask.fits'
hp.write_map(outfn,hp.ud_grade(pl_mask_apod,Nside),dtype='f4',\
             coord='C',overwrite=True)