# Global parameters used throughout.
# NSIDE : int, healpix NSIDE used for mapmaking
# COORD : str, coordinate system
# LEDGES: list, edges of ell-bins for power spectrum measurements
import numpy as np
NSIDE  = 2048
COORD  = 'c'
LEDGES = [10,20,44,79,124,178,243,317,401,495]
LEDGES = LEDGES + list((np.linspace(600**0.5,(3*NSIDE)**0.5,30,endpoint=True)**2).astype(int))