import numpy as np
import sys
import json
from cobaya.likelihood import Likelihood
from scipy.interpolate import interp1d
sys.path.append('../')
from theory.limber               import limb 
from theory.pkCodes              import pmmHEFT,pgmHEFT,pggHEFT
from theory.background           import classyBackground
from likelihoods.gaussLikeSimple import gaussLike
from likelihoods.pack_data_v2    import pack_cl_wl,pack_cov,pack_dndz

class XcorrLike(Likelihood):
    ## From yaml file
    # .json file input (cl's, window functions and covariances)
    jsonfn:   str
    # name of the CMB lensing map 
    kapNames: list
    # names of the galaxy samples
    galNames: list
    # redshift distribution filenames (one for each galaxy sample)
    dndzfns:  list
    # scale cuts
    amin:     list
    amax:     list
    xmin:     list
    xmax:     list
    # fiducial shot noise
    fidSN:    list
    # prior on shot noise (sigma = fidSN*snfrac)
    snfrac:   float
    # fiducial alpha_auto and priors
    fida0:    list
    a0prior:  list
    # fiducial alpha_cross and priors
    fidaX:    list
    aXprior:  list
    # "Chen prior"
    # when this is true alpha_x = alpha_a/(2*b^E_1) + epsilon
    # and the fidaX, aXprior are the fiducial values and priors
    # on epsilon (rather than alphaX)
    chenprior: bool
    # Jeffreys prior on linear parameters
    jeffreys: bool
    # maximize or sample?
    maximize:  bool
    def initialize(self):
        """Sets up the class."""
        self.nsamp = len(self.galNames) # number of galaxy samples
        self.nkap  = len(self.kapNames) # number of CMB lensing maps
        # load the data (stack data [including dndz's] and apply scale cuts)
        self.loadData()
        # Fiducial cosmological (and bias) parameters used to compute eff redshifts 
        fid_cosmo = [0.022,0.1202,0.9667,3.045,67.27,0.06] # omb,omc,ns,ln(1e10 As),H0,Mnu
        fid_bias  = [0.9,0.,0.]                            # b1, b2, bs
        fid = np.array(fid_cosmo+fid_bias)
        # set up the theory prediction class.
        self.clPred = limb(self.dndz, fid, pgmHEFT, pggHEFT, pmmHEFT, classyBackground, zmin=0.001, zmax=1.8, Nz=80)
        # set up the gaussian likelihood class.
        # requires (Gaussian = [mu,sigma]) priors on our three templates 
        # (for each galaxy sample) which are analytically marginalized over.
        # The templates have coefficients: alpha_auto, (2D projected) shot noise, alpha_cross
        def template_priors(isamp):
            a0 = self.fida0[isamp] ; a0p = self.a0prior[isamp]
            SN = self.fidSN[isamp] ; SNp = self.snfrac*SN
            aX = self.fidaX[isamp] ; aXp = self.aXprior[isamp]
            return [[a0,a0p],[SN,SNp],[aX,aXp]]
        tmp_priors = template_priors(0)
        for i in range(1,self.nsamp): tmp_priors += template_priors(i)
        self.tmp_priors = tmp_priors
        print('Using template priors =',tmp_priors)
        self.glk = gaussLike(self.data, self.cov, tmp_priors=np.array(tmp_priors), jeffreys=self.jeffreys)
        
    def get_requirements(self):
        """What we require."""
        reqs = {'omega_b':None,'omega_cdm':None,'n_s':None,'ln1e10As':None,'H0':None,'m_ncdm':None}
        # Build the parameter names we require for each galaxy sample.
        for suf in self.galNames:
            for pref in ['b1','b2','bs','smag']:
                reqs[pref+'_'+suf] = None
        return reqs
        
    def logp(self,**params_values):
        """Return the log-likelihood."""
        if self.maximize: return self.glk.maxLogLike(self.compute_full())
        return self.glk.margLogLike(self.compute_full())
        
    def loadData(self):
        """
        Load the data from json file, stack and apply scale cuts.
        Also load window functions and make dndz matrix.
        """
        # load the json file containing cl's, window functions, and covariances
        with open(self.jsonfn) as outfile:
            jsondata = json.load(outfile)
        self.wla,self.wlx,self.data = pack_cl_wl(jsondata,self.kapNames,self.galNames,self.amin,self.amax,self.xmin,self.xmax)
        self.cov  =                   pack_cov(  jsondata,self.kapNames,self.galNames,self.amin,self.amax,self.xmin,self.xmax)
        dndzs     = [np.loadtxt(self.dndzfns[i]) for i in range(self.nsamp)]
        self.dndz = pack_dndz(dndzs)
        self.pixwin = np.array(jsondata['pixwin'])

    def get_cosmo_parameters(self):
        pp  = self.provider
        omb = pp.get_param('omega_b')
        omc = pp.get_param('omega_cdm')
        ns  = pp.get_param('n_s')
        As  = pp.get_param('ln1e10As')
        H0  = pp.get_param('H0')
        Mnu = pp.get_param('m_ncdm')
        return omb,omc,ns,As,H0,Mnu

    def get_nuisance_parameters(self,i):
        pp   = self.provider
        suf  = self.galNames[i]
        b1   = pp.get_param('b1_'+suf)
        b2   = pp.get_param('b2_'+suf)
        bs   = pp.get_param('bs_'+suf)
        smag = pp.get_param('smag_'+suf)
        return b1,b2,bs,smag
        
    def compute_full(self):
        """
        Do the full prediction (including [pixel] window functions)
        Returns a table with coefficients
        # (1, alpha_a(z1), SN(z1), alpha_x(z1), alpha_a(z2), SN(z2), alpha_x(z2), ...)
        """
        omb,omc,ns,As,H0,Mnu = self.get_cosmo_parameters()
        full_pred = []
        Nls       = []
        for i,suf in enumerate(self.galNames):
            b1,b2,bs,smag = self.get_nuisance_parameters(i)
            params = np.array([omb,omc,ns,As,H0,Mnu,b1,b2,bs])
            # Cgg and Ckg are tables of shape (nell,4)
            # where the four columns correspond to 
            # 1, alpha_auto, shot noise, alpha_cross
            Cgg,Ckg = self.clPred.computeCggCkg(i,params,smag)
            if self.chenprior:
                Cgg[:,1] += Cgg[:,3]/(2.*(1.+b1))
                Ckg[:,1] += Ckg[:,3]/(2.*(1.+b1))
            Nkg = Ckg.shape[0] ; Ngg = Cgg.shape[0]
            # correct for pixel window function
            # shot noise is left untouched
            pixwin_idxs = [0,1,3] 
            for idx in pixwin_idxs:
                Ckg[:,idx] = Ckg[:,idx]*self.pixwin[:Nkg]
                Cgg[:,idx] = Cgg[:,idx]*self.pixwin[:Ngg]**2
            # multiply by the "mask window"
            wa     = self.wla[i][:,:Ngg]
            Cggkgs = np.dot(wa,Cgg)
            for j in range(self.nkap):
                wx     = self.wlx[j][i][:,:Nkg]
                Cggkgs = np.concatenate((Cggkgs,np.dot(wx,Ckg)))
            # stack the data vector
            Nl,Nmon = Cggkgs.shape
            res = np.zeros((Nl,1+(Nmon-1)*self.nsamp))
            res[:,0] = Cggkgs[:,0]
            res[:,1+i*(Nmon-1):1+(i+1)*(Nmon-1)] = Cggkgs[:,1:]
            full_pred.append(res)
        return np.concatenate(full_pred)
    
    def best_fit_raw(self, i, pixwin=True, return_tables=False):
        """
        Returns raw (unbinned) theory prediction with linear
        parameters fixed to their best-fit values.
        """
        omb,omc,ns,As,H0,Mnu = self.get_cosmo_parameters()
        b1,b2,bs,smag = self.get_nuisance_parameters(i)
        params  = np.array([omb,omc,ns,As,H0,Mnu,b1,b2,bs])
        Cgg,Ckg = self.clPred.computeCggCkg(i,params,smag)
        if self.chenprior:
            Cgg[:,1] += Cgg[:,3]/(2.*(1.+b1))
            Ckg[:,1] += Ckg[:,3]/(2.*(1.+b1))
        Nkg = Ckg.shape[0] ; Ngg = Cgg.shape[0]
        pixwin_idxs = [0,1,3]
        for idx in pixwin_idxs:
            if pixwin:
                Ckg[:,idx] = Ckg[:,idx]*self.pixwin[:Nkg]
                Cgg[:,idx] = Cgg[:,idx]*self.pixwin[:Ngg]**2
        tmp_prm_star = self.glk.getBestFitTemp(self.compute_full())
        Ntmp = Cgg.shape[1]-1
        tmp_prm_star = tmp_prm_star[i*Ntmp:(i+1)*Ntmp]
        monomials    = np.array([1.]+list(tmp_prm_star))
        if return_tables: return tmp_prm_star,Cgg,Ckg
        return np.dot(Cgg,monomials),np.dot(Ckg,monomials)

    def best_fit(self):
        """
        Returns theory prediction with all linear 
        parameters fixed to their best-fit values.
        """
        return self.glk.getBestFit(self.compute_full())
    
    def interpolate_Cgigj(self,I,J,alpha_a,alpha_x,fill_value='extrapolate'):
        """
        Interpolates the galaxy cross-spectra given some fiducial evolution
        for the counterterms.
        I      : int
        J      : int
        alpha_a: (nsamp) ndarray, the auto counterterm evaluated at each eff z
        alpha_x: (nsamp) ndarray, the cross counterterm evaluated at each eff z
        """
        omb,omc,ns,As,H0,Mnu = self.get_cosmo_parameters()
        nuisance  = np.array([list(self.get_nuisance_parameters(i)) for i in range(self.nsamp)])
        b1_interp = interp1d(self.clPred.zeff,nuisance[:,0],bounds_error=False,fill_value=fill_value)
        b2_interp = interp1d(self.clPred.zeff,nuisance[:,1],bounds_error=False,fill_value=fill_value)
        bs_interp = interp1d(self.clPred.zeff,nuisance[:,2],bounds_error=False,fill_value=fill_value)
        sm_interp = interp1d(self.clPred.zeff,nuisance[:,3],bounds_error=False,fill_value=fill_value)
        aa_interp = interp1d(self.clPred.zeff,alpha_a,      bounds_error=False,fill_value=fill_value)
        ax_interp = interp1d(self.clPred.zeff,alpha_x,      bounds_error=False,fill_value=fill_value)
        mon_auto  = lambda z: np.array([aa_interp(z)])
        mon_cros  = lambda z: np.array([ax_interp(z)])
        thy_args  = lambda z: np.array([omb,omc,ns,As,H0,Mnu,b1_interp(z),b2_interp(z),bs_interp(z)])
        return self.clPred.computeCgigjZevolution(I,J,thy_args,mon_auto,mon_cros,sm_interp,ext=3)
