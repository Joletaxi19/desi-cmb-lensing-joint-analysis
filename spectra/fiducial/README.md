To get the fiducial Cgg's and Ckg's from `cls_LRGxPR4_bestFit.txt` run the following:

```
import numpy as np

nSide = 2048
nEll  = 3*nSide
cij   = np.loadtxt('cls_LRGxPR4_bestFit.txt').reshape((5,5,nEll))

# cij is a three dimensional array. The last dimension corresponds
# to ell = 0,...,3*nSide-1. cij is symmetric in the first two arguments, 
# which correspond to the basis [kappa, LRGz1, LRGz2, LRGz3, LRGz4].
#
# Explicitly:
#      cij[0,0,:] is a fiducial Ckk calculated with CLASS (w/o recon noise)
#      cij[i,i,:] is C^{g_i g_i} INCLUDING SHOT NOISE with z = zi (i=1,2,3,4) 
#      cij[0,i,:] is C^{k   g_i} with z = zi (i=1,2,3,4)
#
# The remaining entries are the cross-correlations between the 
# different galaxy samples. e.g. cij[1,2,:] is the cross-correlation
# of the galaxies in the first redshift bin with those in the second
# C^{g_1 g_2}. The curves for C^{g_i g_i}, C^{k g_i} are HEFT best-fit 
# predictions. The curves for the galaxy-cross spectra are a HEFT 
# prediction made by linearly interpolating the best-fit nuisance terms 
# from the C^{g_i g_i} fits.
#
# Note that the pixel window function is included, and that we have included
# a filtering in the C^{k g_i} predictions. Explicitly this filter is 
#                    filter = np.exp(-(l/2500)**6)
```

The best-fit projected shot noises for the galaxy samples are `4.01e-6, 2.24e-6, 2.08e-6, 2.31e-6` respectively.