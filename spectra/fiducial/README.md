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
#      cij[0,0,:] is the fiducial Ckk+Nkk (from Planck PR4 release)
#      cij[i,i,:] is C^{g_i g_i} with z = zi (i=1,2,3,4)
#      cij[0,i,:] is C^{k   g_i} with z = zi (i=1,2,3,4)
#
# The remaining entries are the cross-correlations between the 
# different galaxy samples. e.g. cij[1,2,:] is the cross-correlation
# of the galaxies in the first redshift bin with those in the second
# C^{g_1 g_2}.
#
# For ell<1000, the curves for C^{g_i g_i} and C^{k g_i} correspond to a 
# HEFT best-fit, and for ell>1000 I'm using a polynomial fit to the data.
# The curves for C^{g_i g_j} with i \neq j are polynomial fits. All curves
# are set to zero for ell>3000.
```