import numpy  as np
import healpy as hp

def grf_induction_step(alms,cij):
    """
    Given N alms drawn from covariance cij[:N,:N,:], produce a new alm
    with auto-correlation cij[-1,-1,:] and cross-correlations
    cij[-1,:N,:] with the input alms.
    
    alms : a list of N alms, all with the same nell
    cij  : a (N+1,N+1,nell) ndarray, representing the covariance matrix
    """
    N      = len(alms)
    nell   = cij.shape[-1]
    cijsub = cij[:N,:N,:]
    cross  = cij[-1,:N,:]
    # to avoid singular matrices, I'm doing a dumb loop
    cinv   = np.zeros_like(cij)
    for l in range(nell):
        try:
            cinv[:,:,l] = np.linalg.inv(cij[:,:,l])
        except:
            continue
    # compute coefficients for input maps
    coeff  = np.zeros((N,nell))
    # should replace this with something more clever, such as einsum...but for now
    for i in range(N): 
        for j in range(N): 
            coeff[i,:] += cinv[i,j,:]*cross[j,:]
    # add the correlated piece to the result
    result = np.zeros_like(alms[0])
    for i in range(N): result += hp.almxfl(alms[i],coeff[i,:])
    # and the uncorrelated piece
    cdiff = np.zeros(nell)
    for i in range(N):
        for j in range(N):
            cdiff += cross[i,:]*cinv[i,j,:]*cross[j,:]
    cee = cij[-1,-1,:] - cdiff
    eps = hp.synalm(cee)
    result = result + eps
    return result

def gen_maps_from_input_map(imap,cij):
    """
    cij = (N+1,N+1,nell) matrix
    Generate N maps from an initial map (imap) with the 
    appropriate correlations.
    """
    nside = int((len(imap)/12)**0.5)
    N     = cij.shape[0]-1
    ialm  = hp.map2alm(imap,use_pixel_weights=True)
    oalm  = [ialm]
    for i in range(N):
        cijsub = cij[:i+2,:i+2,:]
        nalm = grf_induction_step(oalm,cijsub)
        oalm = oalm + [nalm]
    omaps = [hp.alm2map(alm,nside) for alm in oalm]
    return omaps