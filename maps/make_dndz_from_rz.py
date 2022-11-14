#!/usr/bin/env python3
#
# Generate the dN/dz files in the format we need from Rongpu's input files.
# The individual north and south dN/dz's are weighted by the relative areas.
# The code prints some helpful statistics to stdout.
#
import numpy as np
from scipy.integrate import simps


db   = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/daily_v0.1/'
fn   = 'main_lrg_pz_dndz_daily_v0.1_dz_0.01.txt'
dndz = np.loadtxt(db + fn)
zcen = 0.5*(dndz[:,0]+dndz[:,1]) # Take "z" as the bin center.
# Use this to "blind" the analysis:
# a,b  = 1+0.01*np.random.normal(size=1),0.02*np.random.normal(size=1)
# zcen = a*zcen + b
#
# Set the north/south weights, using fraction of total area.
areas = np.array([4213.,12443.])
areas/= areas.sum()
#
print("# {:>4s} {:>8s} {:>8s}".format("samp","zbar","delz"))
for isamp in [1,2,3,4]:
    # Select a weighted sum of the north and south dN/dz.
    fz = areas[0]*dndz[:,isamp+2]+areas[1]*dndz[:,isamp+7]
    # Only write out values with more than 1% of the peak.
    ww = np.nonzero( fz>0.01*np.max(fz) )[0]
    with open("lrg_s{:02d}_dndz.txt".format(isamp),"w") as fout:
        fout.write("# {:>8s} {:>10s}\n".format("z","dN/dz"))
        for i in ww:
            fout.write("{:10.4f} {:10.6f}\n".format(zcen[i],fz[i]))
    #
    zbar = simps(zcen   *fz,x=zcen)/simps(fz,x=zcen)
    delz = simps(zcen**2*fz,x=zcen)/simps(fz,x=zcen)
    delz = np.sqrt( delz-zbar**2 )
    print("{:6d} {:8.2f} {:8.4f}".format(isamp,zbar,delz))
    #
