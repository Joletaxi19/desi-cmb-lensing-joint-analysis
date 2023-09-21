# This file makes a DES mask and stores it as des_mask.fits (2048 HEALPiX map in celestial coordinates)
# The mask is defined by having a positive "detection fraction" in any of the g,r,i,z,Y bands.

NSIDE_OUT = 2048

import numpy as np
import urllib.request
import os
import healpy as hp

bands   = ['g','r','i','z','Y']
website = 'https://desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/Coverage_DR2/'
fnames  = [f'dr2_hpix_4096_frac_detection_{b}.fits.fz' for b in bands]

# fetch detection fraction files from the web (HEALPiX maps in celestial coords)
# and save them in the current directory
for fn in fnames: urllib.request.urlretrieve(website+fn, fn)
maps = [hp.read_map(fn) for fn in fnames]
for i in range(len(maps)): maps[i][np.where(maps[i]<=0)]=0.
tot = hp.ud_grade(np.sum(maps,axis=0),NSIDE_OUT)
tot[np.where(tot>0.)]=1.
hp.write_map('masks/des_mask.fits',tot,overwrite=True)
# delete detection fraction files
for fn in fnames: os.remove(fn)