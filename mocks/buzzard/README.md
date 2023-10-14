# Buzzard mocks

Power spectra and window functions: `lrg_s1i_XXX.txt` where `XXX = cls, wla, wlx` and `i=1,2,3,4` for `z = z1,z2,z3,z4`.

Redshift distribution: `combined_dNdz.txt`. First column is redshifts, remaining four are dNdz for `z = z1,z2,z3,z4`.

Covariance for PR4: `lrg_pr4_multi_cov.txt`. Basis is (Cgg1, Ckg1, Cgg2, Ckg2, ... Ckg4) with 20 ell bins (same bins as `*cls.txt`).

### Fiducial cosmology
```
# inputs
n_s       = 0.96 
h         = 0.7 
A_s       = 2.145e-09 
omega_b   = 0.02254 
omega_cdm = 0.1176

# derived
sigma8         = 0.82530
100*theta_star = 1.0476264
```

### Magnification biases

For each redshift bin: `1.062, 0.973, 0.825 0.8`

### Projected shot noise

For each redshift bin: `3.5837e-06, 1.9653e-06, 1.8574e-06, 2.1008e-06`