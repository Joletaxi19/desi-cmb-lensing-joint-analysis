# Makes subcatalogs from the full Websky halo catalog (halos.pksc).
# This piece of code is largely based off of:
# https://mocks.cita.utoronto.ca/data/websky/v0.0/readhalos.py 

import healpy as hp
from   cosmology import *

rho = 2.775e11*omegam*h**2 # Msun/Mpc^3

f=open('halos.pksc')
N=np.abs(np.fromfile(f,count=3,dtype=np.int32)[0]) # added abs (not sure why negative?)

print(N,'total halos')

print('Started loading catalog')
catalog=np.fromfile(f,count=N*10,dtype=np.float32)
catalog=np.reshape(catalog,(N,10))
print('Finished loading catalog!')

x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
R  = catalog[:,6] # Mpc

# convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
M200m    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
redshift = zofchi(chi)      

theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians

def make_subcatalog(z1,z2,s):
    I_lrg = np.where((redshift>z1) & (redshift<z2))
    N_lrg = len(I_lrg[0])
    print(N_lrg,'LRGs in sample',str(s))
    LRG_catalog = np.zeros((N_lrg,4))
    LRG_catalog[:,0] = redshift[I_lrg]
    LRG_catalog[:,1] =    theta[I_lrg]
    LRG_catalog[:,2] =      phi[I_lrg]
    LRG_catalog[:,3] =    M200m[I_lrg]
    LRG_catalog = LRG_catalog.flatten()
    LRG_catalog.astype('float32').tofile('LRG_halos_'+'s'+str(s)+'.pksc')

make_subcatalog(0.25,0.75,1)
make_subcatalog(0.30,0.90,2)
make_subcatalog(0.50,1.10,3)
make_subcatalog(0.60,1.40,4)
