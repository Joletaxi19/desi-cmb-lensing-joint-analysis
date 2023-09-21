# This file downloads the 40% and 60% galactic masks from the PLA and saves
# them as gal40_mask.fits and gal60_mask.fits in equatorial coordinates
# with nSide=2048

import numpy as np
import urllib.request
import os
import healpy as hp

# download from web
website = 'http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID='
fname   = 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
urllib.request.urlretrieve(website+fname, fname)

# rotate from galactic to equatorial coords
# and write to .fits files
msk40_gal = hp.read_map(fname,field=1)
msk60_gal = hp.read_map(fname,field=2)
theta_gal,phi_gal = hp.pix2ang(2048,np.arange(len(msk40_gal)))
theta_cel,phi_cel = hp.rotator.Rotator(coord='cg')(theta_gal, phi_gal)
pix = hp.ang2pix(2048,theta_cel,phi_cel)
msk40 = np.zeros_like(msk40_gal)
msk40 = msk40_gal[pix]
msk60 = np.zeros_like(msk60_gal)
msk60 = msk60_gal[pix] 
hp.write_map('masks/gal40_mask.fits',msk40,overwrite=True)
hp.write_map('masks/gal60_mask.fits',msk60,overwrite=True)

# delete the file from the web
os.remove(fname)