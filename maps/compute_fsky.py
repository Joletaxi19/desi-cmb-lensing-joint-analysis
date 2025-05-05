import healpy as hp
import numpy as np

pr4_mask = hp.read_map('masks/PR4_lens_mask.fits')
lrg_masks = [hp.read_map(f'masks/lrg_s0{i}_msk.hpx2048.public.fits.gz') for i in range(1,5)]
mask_lrg = np.clip(sum(lrg_masks), 0, 1)
mask_tot = (pr4_mask>0) & (mask_lrg>0)

f_sky_tot = np.mean(mask_tot)
print("f_sky total :", f_sky_tot)