import numpy  as np
import healpy as hp
from   astropy.table import Table
import urllib.request
import os

def make_aux_masks(NSIDE_OUT=2048, COORD_OUT='c', outdir='masks', deccuts=[-15,-30], 
                   ebvcuts=[0.05,0.1,0.15], starcuts=[2500,1500], verbose_sffx=False):
    """
    Makes the following (binary) masks
        ngc_mask.fits          | NGC 
        sgc_mask.fits          | SGC 
        north_mask.fits        | (DEC > 32.375) and NGC
        des_mask.fits          | DES is defined as everywhere with positive "detection 
                                 fraction" in any of the DES DR2 g,r,i,z,Y bands.
        decals_mask.fits       | (not North) and (not DES) and (DEC > -15)
        dec[P or M]#_mask.fits | creates a mask for each # [integer] in deccuts that 
                                 is defined by DEC <= \PM #
        ebv_#_mask.fits        | creates a mask for each # [float] in ebvcuts that is 
                                 is defined by EBV <= #
        gal#_mask.fits         | Planck 60% and 40% galactic masks
        star_#_mask.fits       | creates a mask satisfying stellar-density<# for each
                                 # in starcuts [stars per square degree]
    at a given NSIDE_OUT and COORD_OUT system, and saves them to outdir/. If 
    verbose_sffx=True, appends the string "_cord.{COORD_OUT}_nside.{NSIDE_OUT}" 
    to the end of each mask name.
    """
    
    if not os.path.exists(outdir): os.mkdir(outdir)
    sffx = '' if (not verbose_sffx) else f'_cord.{COORD_OUT}_nside.{NSIDE_OUT}'

    # Make DEC and RA in celestial coords and 
    # rotate to COORD_OUT coords
    npix      = 12*NSIDE_OUT**2
    theta,phi = hp.pix2ang(NSIDE_OUT,np.arange(npix))
    DEC_,RA_  = 90-np.degrees(theta),np.degrees(phi)
    rot = hp.rotator.Rotator(coord=f'c{COORD_OUT}')
    DEC = rot.rotate_map_pixel(DEC_)
    RA  = rot.rotate_map_pixel(RA_)

    ## Make NGC/SGC masks in galactic coords 
    ## and rotate to COORD_OUT coords
    ngc_mask = np.ones(npix) ; ngc_mask[np.where(DEC_<=0.)] = 0.
    sgc_mask = np.ones(npix) ; sgc_mask[np.where(DEC_>0.)]  = 0.
    rot      = hp.rotator.Rotator(coord=f'g{COORD_OUT}')
    ngc_mask = np.round(rot.rotate_map_pixel(ngc_mask))
    sgc_mask = np.round(rot.rotate_map_pixel(sgc_mask))
    hp.write_map(f'{outdir}/ngc_mask{sffx}.fits',ngc_mask,overwrite=True,dtype=np.int32)
    hp.write_map(f'{outdir}/sgc_mask{sffx}.fits',sgc_mask,overwrite=True,dtype=np.int32)

    ## Make "North" mask in COORD_OUT coords
    north_mask = ngc_mask.copy()
    north_mask[np.where(DEC<=32.375)] = 0.
    hp.write_map(f'{outdir}/north_mask{sffx}.fits',north_mask,overwrite=True,dtype=np.int32)

    ## Make DES mask in COORD_OUT coords
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
    # rotate to COORD_OUT coordinates
    rot = hp.rotator.Rotator(coord=f'c{COORD_OUT}')
    des_mask = np.round(rot.rotate_map_pixel(des_mask))
    hp.write_map(f'{outdir}/des_mask{sffx}.fits',des_mask,overwrite=True,dtype=np.int32)
    # delete detection fraction files
    for fn in fnames: os.remove(fn)

    ## Make DECaLS mask in COORD_OUT coords
    decals_mask = np.ones(npix) - north_mask - des_mask
    decals_mask[np.where(DEC<=-15)] = 0.
    hp.write_map(f'{outdir}/decals_mask{sffx}.fits',decals_mask,overwrite=True,dtype=np.int32)
    
    ## Make DEC <= DECcuts masks in COORD_OUT coords
    for cut in np.array(deccuts,dtype=np.int32):
        mask = np.ones(npix)
        mask[np.where(DEC>cut)] = 0.
        sgn = 'p' if cut>=0 else 'm'
        hp.write_map(f'{outdir}/DEC{sgn}{np.abs(cut)}_mask{sffx}.fits',mask,overwrite=True,dtype=np.int32)
        
    ## Make EBV < X masks in COORD_OUT coords
    version   = 0
    colors    = 'rz'
    nside     = 256
    # fetch Rongpu's EBV map, which has nside=256 and is in celestial coords
    bd        = f'/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/v{version}/kp3_maps/'
    ebv_table = Table.read(bd+f'v{version}_desi_ebv_{colors}_{nside}.fits')
    rot       = hp.rotator.Rotator(coord=f'c{COORD_OUT}')
    ebv_sfd   = rot.rotate_map_pixel(hp.ud_grade(ebv_table['EBV_SFD'],NSIDE_OUT))
    for cut in np.array(ebvcuts):
        mask = np.ones(npix)
        mask[np.where(ebv_sfd>cut)] = 0.
        hp.write_map(f'{outdir}/ebv_{cut:0.2f}_mask{sffx}.fits',mask,overwrite=True,dtype=np.int32)
        
    # Fetch 40% and 60% galactic masks from PLA  
    website = 'http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID='
    fname   = 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'
    urllib.request.urlretrieve(website+fname, fname)
    # rotate from galactic to COORD_OUT coords
    # and write to .fits files
    msk40_gal = hp.read_map(fname,field=1)
    msk60_gal = hp.read_map(fname,field=2)
    rot       = hp.rotator.Rotator(coord=f'g{COORD_OUT}')
    msk40     = np.round(rot.rotate_map_pixel(hp.ud_grade(msk40_gal,NSIDE_OUT)))
    msk60     = np.round(rot.rotate_map_pixel(hp.ud_grade(msk60_gal,NSIDE_OUT)))
    hp.write_map(f'{outdir}/gal40_mask{sffx}.fits',msk40,overwrite=True,dtype=np.int32)
    hp.write_map(f'{outdir}/gal60_mask{sffx}.fits',msk60,overwrite=True,dtype=np.int32)
    # delete the file from the web
    os.remove(fname)
    
    # Make stellar density masks
    release   = 'dr9'
    version   = '1.0.0'
    db        = '/global/cfs/cdirs/desi/target/catalogs/'
    db       += release+'/'+version+'/pixweight/main/resolve/dark/'
    fn        = db+'pixweight-1-dark.fits'
    pxw_nside = 256
    pxw_nest  = True
    pxw       = Table.read(fn)
    stars_nside = 64
    stars     = hp.ud_grade(pxw['STARDENS'],stars_nside,order_in='NEST',order_out='RING')
    theta,phi = hp.pix2ang(NSIDE_OUT,np.arange(12*NSIDE_OUT**2))
    ipix      = hp.ang2pix(stars_nside,theta,phi)
    rot       = hp.rotator.Rotator(coord=f'c{COORD_OUT}')
    for cut in np.array(starcuts):
        mask  = np.ones(12*NSIDE_OUT**2)
        mask[stars[ipix]>cut] = 0.
        mask  = np.round(rot.rotate_map_pixel(mask))
        hp.write_map(f'{outdir}/star_{int(cut)}_mask{sffx}.fits',mask,overwrite=True,dtype=np.int32)
    

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from globe import NSIDE,COORD
    make_aux_masks(NSIDE_OUT=NSIDE,COORD_OUT=COORD)
