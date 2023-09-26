# Takes Rongpu's file and makes it slightly more "digestable" for our theory+likelihood code
import numpy as np

# areas used when weighting regions
fsky_north  = 0.113
fsky_decals = 0.204
fsky_des    = 0.123
fsky_south  = fsky_decals+fsky_des

# generic header
header  = 'Angular number densities per z bin ("number per bin")\n'
header += 'i.e., the number of galaxies per sq.deg. that have zmin<z<zmax.\n'
header += 'Both imaging and spectroscopic weights are included.\n\n'
header += 'Columns are: redshift, dNdz (sample 1), dNdz (sample 2), dNdz (sample 3), dNdz (sample 4)'

# file location
version = '0.4'
bdir    = f'/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v{version}/'

# north, south and full sample
fname   = f'main_lrg_pz_dndz_iron_v{version}_dz_0.01.txt'
xxx = np.genfromtxt(bdir+fname)
z = (xxx[:,0]+xxx[:,1])/2
north = np.array([z,xxx[:,8],xxx[:,9],xxx[:,10],xxx[:,11]]).T
south = np.array([z,xxx[:,13],xxx[:,14],xxx[:,15],xxx[:,16]]).T
full  = (north*fsky_north+south*fsky_south)/(fsky_north+fsky_south)
np.savetxt('LRG_dNdz_north.txt',north,header=header)
np.savetxt('LRG_dNdz_south.txt',north,header=header)
np.savetxt('LRG_dNdz.txt'      ,full ,header=header)

# DECaLS and DES regions individually
fname   = f'main_lrg_pz_ngal_decals_des_iron_v{version}_dz_0.02.txt'
xxx = np.genfromtxt(bdir+fname)
z = (xxx[:,0]+xxx[:,1])/2
decals = np.array([z,xxx[:,3],xxx[:,4],xxx[:,5],xxx[:,6]]).T
des = np.array([z,xxx[:,8],xxx[:,9],xxx[:,10],xxx[:,11]]).T
np.savetxt('LRG_dNdz_decals.txt',decals,header=header)
np.savetxt('LRG_dNdz_des.txt',des,header=header)