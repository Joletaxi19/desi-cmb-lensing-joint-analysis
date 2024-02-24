from do_mc_corr import *
import sys
sys.path.append('../')
from globe import NSIDE

job = int(sys.argv[1])

# load some masks (which may or may not be used depending on the job)
isamp    = 1
bdir     = '/pscratch/sd/m/mwhite/DESI/MaPar/maps/'
lrg_mask = hp.read_map(bdir+f'lrg_s0{isamp}_msk.hpx2048.fits')
north    = hp.read_map('../maps/masks/north_mask.fits')
des      = hp.read_map('../maps/masks/des_mask.fits')
decals   = hp.read_map('../maps/masks/decals_mask.fits')
PR3mask  = hp.read_map(f'../maps/masks/PR3_lens_mask.fits')
PR4mask  = hp.read_map(f'../maps/masks/PR4_lens_mask.fits')

# baseline LRG mask ("full") correlated with act dr6
def do_dr6(lrg_mask,lrg_name,option='baseline'):
    release = 'dr6_lensing_v1'
    bdir    = f'/global/cfs/projectdirs/act/www/{release}/'
    dr6_mask= hp.ud_grade(hp.read_map(f'{bdir}maps/{option}/mask_act_dr6_lensing_v1_healpix_nside_4096_{option}.fits'),NSIDE)
    make_mc_cls(f'lrg-full-z{isamp}',lrg_mask,dr6_mask,'c',lensmap='DR6',option=option)
    
if job==0: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='baseline')
if job==1: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='cibdeproj')
if job==2: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='f090')
if job==3: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='f090_tonly')
if job==4: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='f150')
if job==5: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='f150_tonly')
if job==6: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='galcut040')
if job==7: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='galcut040_polonly')
if job==8: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='polonly')
if job==9: do_dr6(lrg_mask,f'lrg-full-z{isamp}',option='tonly')

# baseline LRG mask ("full") correlated with PR3
if job==10: make_mc_cls(f'lrg-full-z{isamp}',lrg_mask,PR3mask,'c',lensmap='PR3')

# different LRG masks correlated with PR4
if job==11: make_mc_cls(f'lrg-full-z{isamp}'  ,lrg_mask,       PR4mask,'c',lensmap='PR4')
if job==12: make_mc_cls(f'lrg-north-z{isamp}' ,lrg_mask*north, PR4mask,'c',lensmap='PR4')
if job==13: make_mc_cls(f'lrg-decals-z{isamp}',lrg_mask*decals,PR4mask,'c',lensmap='PR4')
if job==14: make_mc_cls(f'lrg-des-z{isamp}'   ,lrg_mask*des,   PR4mask,'c',lensmap='PR4')