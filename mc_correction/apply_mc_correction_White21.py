import numpy as np
from do_mc_corr import bin_mc_corr

mc_corr = bin_mc_corr('LRG_full_z1_PR3')
print('ell    mc_corr')
print(mc_corr)
mc_corr = mc_corr[:,1]

for i in range(4):
    dat = np.genfromtxt(f'../data/White21_data/lrg_s0{i+1}_cls.txt').copy()
    dat[:,2] *= mc_corr
    np.savetxt(f'../data/White21_data/lrg_s0{i+1}_cls_mccorr.txt',dat)
    
