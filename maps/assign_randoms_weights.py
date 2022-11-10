#
import numpy as np
import warnings
import yaml

#from astropy.table        import Table, vstack, hstack
#from sklearn.linear_model import LinearRegression


def get_randoms_weights(randoms, weights_path, bin_index):
    with open(weights_path, "r") as f:
        linear_coeffs = yaml.safe_load(f)
    # Assign zero weights to randoms with invalid imaging properties
    mask_bad = np.full(len(randoms), False)
    xnames_fit = list(linear_coeffs['south_bin_1'].keys())
    xnames_fit.remove('intercept')
    for col in xnames_fit:
        mask_bad |= ~np.isfinite(randoms[col])
    if np.sum(mask_bad)!=0:
        print('{} invalid randoms'.format(np.sum(mask_bad)))
    #
    weights = np.zeros(len(randoms))
    #
    for field in ['north','south']:
        if field=='south':
            photsys = 'S'
        elif field=='north':
            photsys = 'N'
        #
        mask  = randoms['PHOTSYS']==photsys
        mask &= (~mask_bad)
        data  = np.column_stack([randoms[mask][xname] for xname in xnames_fit])
        #
        bin_str = '{}_bin_{}'.format(field, bin_index)
        #reg = LinearRegression()
        #reg.intercept_ = linear_coeffs[bin_str]['intercept']
        #reg.coef_ = np.array([linear_coeffs[bin_str][xname] for xname in xnames_fit])
        #weights[mask] = reg.predict(data)
        #
        # Create array of coefficients, with the first coefficient being the intercept
        coeffs = np.array([linear_coeffs[bin_str]['intercept']]+[linear_coeffs[bin_str][xname]\
                 for xname in xnames_fit])
        # create 2-D array of imaging properties, with the first columns being unity
        data1 = np.insert(data, 0, 1., axis=1)
        weights[mask] = np.dot(coeffs, data1.T)
    return(weights)


# Prepare the randoms
#
#min_nobs = 2
#maskbits = sorted([1, 8, 9, 11, 12, 13])
#
#randoms_columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS', 'PHOTSYS',
#                   'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
#                   'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'EBV']
#
#randoms = Table(fitsio.read('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits', columns=randoms_columns))
#
## randoms = Table(fitsio.read('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits', columns=randoms_columns,
##                            rows=np.arange(int(1e6))))
#
#mask = (randoms['NOBS_G']>=min_nobs) & (randoms['NOBS_R']>=min_nobs) & (randoms['NOBS_Z']>=min_nobs)
#randoms = randoms[mask]
#
#mask_clean = np.ones(len(randoms), dtype=bool)
#for bit in maskbits:
#    mask_clean &= (randoms['MASKBITS'] & 2**bit)==0
#randoms = randoms[mask_clean]
#
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    randoms['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_G'])))-9) - 3.214*randoms['EBV']
#    randoms['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_R'])))-9) - 2.165*randoms['EBV']
#    randoms['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_Z'])))-9) - 1.211*randoms['EBV']
#    randoms['psfdepth_w1mag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W1'])))-9) - 0.184*randoms['EBV']
#    randoms['psfdepth_w2mag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W2'])))-9) - 0.113*randoms['EBV']
#
## Get weights
#
#weights_path = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/imaging_weights/main_lrg_linear_coeffs_rw1.yaml'
#bin_index = 5
#
#weights = get_randoms_weights(randoms, weights_path, bin_index)
#
