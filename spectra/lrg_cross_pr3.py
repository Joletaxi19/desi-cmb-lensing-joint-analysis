from calc_cl import *
sys.path.append('../')
sys.path.append('../mc_correction/')
from globe      import LEDGES,NSIDE
from do_mc_corr import apply_mc_corr

# load maps, masks and CMB lensing noise curves
lrg_maps  = [hp.read_map(f'../maps/lrg_s0{isamp}_del.hpx2048.fits') for isamp in range(1,5)]
lrg_masks = [hp.read_map(f'../maps/masks/lrg_s0{isamp}_msk.hpx2048.fits') for isamp in range(1,5)]
pr3_map   = [hp.read_map(f'../maps/PR3_lens_kap_filt.hpx2048.fits')]
pr3_mask  = [hp.read_map(f'../maps/masks/PR3_lens_mask.fits')]
pr3_nkk   = np.loadtxt(f'../data/PR3_lens_nlkk_filt.txt')
fnout     = f'lrg_cross_pr3.json'

# give our maps & masks some names
kapNames = ['PR3']
galNames = ['LRGz1','LRGz2','LRGz3','LRGz4']
names    = kapNames + galNames
msks     = pr3_mask + lrg_masks
maps     = pr3_map  + lrg_maps

# curves for covariance estimation
cij      = np.loadtxt(f'fiducial/cls_LRGxPR4_bestFit.txt').reshape((5,5,3*2048))
ells     = np.arange(cij.shape[-1])
cij[0,0] = np.interp(ells,pr3_nkk[:,0],pr3_nkk[:,2],right=0)

# compute power spectra and window functions, save to json file
pairs = [[0,1],[0,2],[0,3],[0,4],[1,1],[2,2],[3,3],[4,4]]
full_master(LEDGES,maps,msks,names,fnout,cij=cij,do_cov=True,pairs=pairs)

# apply MC correction
bdir = '../mc_correction/sims/'
apply_mc_corr(fnout,fnout,kapNames[0],galNames,[bdir+'lrg-full-z1_PR3-baseline']*4)