import numpy as np

# Takes Rongpu's file and makes it slightly more "digestable"

bdir = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.2/'
fname = bdir + 'main_lrg_pz_dndz_iron_v0.2_dz_0.01.txt'

xxx = np.genfromtxt(fname)
z = (xxx[:,0]+xxx[:,1])/2
dat = np.array([z,xxx[:,3],xxx[:,4],xxx[:,5],xxx[:,6]]).T

header  = 'Angular number densities per z bin ("number per bin")\n'
header += 'i.e., the number of galaxies per sq.deg. that have zmin<z<zmax.\n'
header += 'Values reflect weighted averages of north+south weighted by area\n'
header += '(with the north and south having weights of 1 and 2.2, respectively).\n'
header += 'Both imaging and spectroscopic weights are included.\n'
header += 'The dz size is 0.01.\n\n'
header += 'Columns are: redshift, dNdz (sample 1), dNdz (sample 2), dNdz (sample 3), dNdz (sample 4)'

np.savetxt('LRG_dNdz.txt',dat,header=header)