# This file makes masks for the NGC, SGC, DEC<-15, North, DECaLS and DES "footprints"
# and stores them as ngc_mask.fits, sgc_mask.fits, dec15_mask.fits, north_mask.fits, 
# decals_mask.fits,  des_mask.fits (HEALPix map in in celestial coords). Each of these 
# "footprints" should be multiplied by the appropriate LRG mask (which includes galactic/stellar cuts)

import numpy as np
import healpy as hp
import urllib.request
import os

NSIDE_OUT = 2048
npix      = 12*NSIDE_OUT**2
theta,phi = hp.pix2ang(NSIDE_OUT,np.arange(npix))
DEC,RA    = 90-np.degrees(theta),np.degrees(phi)

### Make NGC/SGC masks
ngc_mask = np.ones(npix)
ngc_mask[np.where(DEC<=0.)] = 0.
sgc_mask = np.ones(npix)
sgc_mask[np.where(DEC>0.)]  = 0.
rot = hp.rotator.Rotator(coord='gc')
ngc_mask = np.round(rot.rotate_map_pixel(ngc_mask))
sgc_mask = np.round(rot.rotate_map_pixel(sgc_mask))
hp.write_map('masks/ngc_mask.fits',ngc_mask,overwrite=True,dtype=np.int32)
hp.write_map('masks/sgc_mask.fits',sgc_mask,overwrite=True,dtype=np.int32)

### Make DEC < -15 mask
dec15_mask = np.ones(npix)
dec15_mask[np.where(DEC>-15.)] = 0.
hp.write_map('masks/dec15_mask.fits',dec15_mask,overwrite=True,dtype=np.int32)

### Make "North" mask
# We define the Northern region as DEC > 32.375 in the NGC
north_mask = ngc_mask.copy()
north_mask[np.where(DEC<=32.375)] = 0.
hp.write_map('masks/north_mask.fits',north_mask,overwrite=True,dtype=np.int32)

### Make DES mask
# The DES mask is defined by having a positive "detection fraction" in any of the g,r,i,z,Y bands.
bands   = ['g','r','i','z','Y']
website = 'https://desdr-server.ncsa.illinois.edu/despublic/dr2_tiles/Coverage_DR2/'
fnames  = [f'dr2_hpix_4096_frac_detection_{b}.fits.fz' for b in bands]
# fetch detection fraction files from the web (HEALPix maps in celestial coords)
# and temporarily save them in the current directory  
for fn in fnames: urllib.request.urlretrieve(website+fn, fn)
maps = [hp.read_map(fn) for fn in fnames]
for i in range(len(maps)): maps[i][np.where(maps[i]<=0)]=0.
des_mask = hp.ud_grade(np.sum(maps,axis=0),NSIDE_OUT)
des_mask[np.where(des_mask>0.)]=1.
hp.write_map('masks/des_mask.fits',des_mask,overwrite=True,dtype=np.int32)
# delete detection fraction files
for fn in fnames: os.remove(fn)

### Make DECaLS mask
decals_mask = np.ones(npix) - north_mask - des_mask
decals_mask[np.where(DEC<=-15)] = 0.
hp.write_map('masks/decals_mask.fits',decals_mask,overwrite=True,dtype=np.int32)