Mock BAO datasets for 6dF and SDSS DR7 & DR12, written in a cobaya-friendly format. The mock data (`.dat` files) are calculated assuming a Buzzard cosmology (below), and do not include any scatter. I assume `rs_fid = 147.78` for all of the mock data. See `yamls/add_mock_bao.yaml` as an example for post-processing. 

Buzzard cosmology:
```
n_s       = 0.96 
h         = 0.7 
A_s       = 2.1186e-09 
omega_b   = 0.02254 
omega_cdm = 0.1176
```