import numpy as np
import sys
from cobaya.likelihood import Likelihood
sys.path.append('../')
from theory.limber               import limb 
from theory.pkCodes              import pmmHEFT,pgmHEFT,pggHEFT
from theory.background           import classyBackground
from likelihoods.gaussLikeSimple import gaussLike

class XcorrLike(Likelihood):
    # From yaml file
    # names for each sample
    suffx:  list
    # Cl's and cov
    clsfn:  list
    covfn:  str
    # window functions
    wlafn:  list
    wlxfn:  list
    # scale cuts
    amin:   list
    amax:   list
    xmin:   list
    xmax:   list
    # redshift distribuiton
    dndzfn: str
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
    # Chen prior
    # when this is true alpha_x = alpha_0/(2*b^E_1) + epsilon
    # and the fidaX, aXprior are the fiducial values and priors
    # on epsilon (rather than alphaX)
    chenprior: bool
    # maximize or sample?
    maximize:  bool
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        fid_cosmo = [0.022,0.1202,0.9667,3.045,67.27,0.06] # omb,omc,ns,ln(1e10 As),H0,Mnu
        fid_bias  = [0.9,0.,0.]                            # b1, b2, bs
        fid = np.array(fid_cosmo+fid_bias)
        self.clPred = limb(self.dndzfn, fid, pgmHEFT, pggHEFT, pmmHEFT, classyBackground, zmin=0.001, zmax=1.8, Nz=80)
        def template_priors(isamp):
            a0 = self.fida0[isamp] ; a0p = self.a0prior[isamp]
            SN = self.fidSN[isamp] ; SNp = self.snfrac*SN
            aX = self.fidaX[isamp] ; aXp = self.aXprior[isamp]
            return [[a0,a0p],[SN,SNp],[aX,aXp]]
        tmp_priors = template_priors(0)
        for i in range(1,len(self.suffx)): tmp_priors += template_priors(i)
        print('Using template priors =',tmp_priors)
        self.glk = gaussLike(self.data, self.cov, tmp_priors=np.array(tmp_priors))
        
    def get_requirements(self):
        """What we require."""
        reqs = {'omega_b':None,'omega_cdm':None,'n_s':None,'ln1e10As':None,'H0':None,'m_ncdm':None}
        # Build the parameter names we require for each sample.
        for suf in self.suffx:
            for pref in ['b1','b2','bs','smag']:
                reqs[pref+'_'+suf] = None
        return(reqs)
        
    def logp(self,**params_values):
        """Return the log-likelihood."""
        if self.maximize: return self.glk.maxLogLike(self.compute_full())
        return self.glk.margLogLike(self.compute_full())
        
    def loadData(self):
        """Load the data, covariance and windows from files."""
        Nsamp = len(self.suffx)
        # load Cl's
        cls = np.array([np.loadtxt(fn) for fn in np.array(self.clsfn)])
        ell = cls[0,:,0]
        # define scale cuts
        #Iacut = [np.where(ell <= lmax)[0] for lmax in np.array(self.acut)]
        #Ixcut = [np.where(ell <= lmax)[0] for lmax in np.array(self.xcut)]  
        Iacut = [np.where((ell<=self.amax[i])&(ell>=self.amin[i]))[0] for i in range(Nsamp)]
        Ixcut = [np.where((ell<=self.xmax[i])&(ell>=self.xmin[i]))[0] for i in range(Nsamp)]
        # stack the data (Cgg1,Ckg1,Cgg2,Ckg2,...)
        # after applying scale cuts
        data = np.array([])
        for i in range(Nsamp): data = np.concatenate((data,cls[i,Iacut[i],1],cls[i,Ixcut[i],2])) 
        self.data = data
        # load the covariance matrix and apply scale cuts
        full_cov = np.loadtxt(self.covfn)
        Icov = []
        for i in range(Nsamp):
            Icov += list(40*i + Iacut[i]) + list(40*i + 20 + Ixcut[i])
            #Icov += list(range(40*i,40*i+len(Iacut[i]))) + list(range(20+40*i,20+40*i+len(Ixcut[i])))
        print('Using these idexes for the covariance matrix',Icov)
        self.cov = full_cov[:,Icov][Icov,:]
        # load window functions and apply scale cuts
        self.wla = []
        self.wlx = []
        for i in range(Nsamp):
            self.wla.append(np.loadtxt(self.wlafn[i])[Iacut[i],:])
            self.wlx.append(np.loadtxt(self.wlxfn[i])[Ixcut[i],:])
            
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
        suf  = self.suffx[i]
        b1   = pp.get_param('b1_'+suf)
        b2   = pp.get_param('b2_'+suf)
        bs   = pp.get_param('bs_'+suf)
        smag = pp.get_param('smag_'+suf)
        return b1,b2,bs,smag
            
    def best_fit_raw(self, i, return_tables=False):
        """
        Returns raw theory prediction with linear
        parameters fixed to their best-fit values.
        """
        omb,omc,ns,As,H0,Mnu = self.get_cosmo_parameters()
        b1,b2,bs,smag = self.get_nuisance_parameters(i)
        params  = np.array([omb,omc,ns,As,H0,Mnu,b1,b2,bs])
        Cgg,Ckg = self.clPred.computeCggCkg(i,params,smag)
        Nkg = Ckg.shape[0] ; Ngg = Cgg.shape[0]
        tmp_prm_star = self.glk.getBestFitTemp(self.compute_full())
        Ntmp = Cgg.shape[1]-1
        tmp_prm_star = tmp_prm_star[i*Ntmp:(i+1)*Ntmp]
        monomials    = np.array([1.]+list(tmp_prm_star))
        if return_tables: return tmp_prm_star,Cgg,Ckg
        return np.dot(Cgg,monomials),np.dot(Ckg,monomials)

    def compute_full(self):
        """
        Do the full prediction (including window function)
        Returns a table with coefficients
        # (1, alpha_a(z1), SN(z1), alpha_x(z1), alpha_a(z2), SN(z2), alpha_x(z2), ...)
        """
        omb,omc,ns,As,H0,Mnu = self.get_cosmo_parameters()
        full_pred = []
        Nls       = []
        for i,suf in enumerate(self.suffx):
            b1,b2,bs,smag = self.get_nuisance_parameters(i)
            params = np.array([omb,omc,ns,As,H0,Mnu,b1,b2,bs])
            Cgg,Ckg = self.clPred.computeCggCkg(i,params,smag)
            if self.chenprior:
                Cgg[:,1] += Cgg[:,3]/(2.*(1.+b1))
                Ckg[:,1] += Ckg[:,3]/(2.*(1.+b1))
            Nkg = Ckg.shape[0] ; Ngg = Cgg.shape[0]
            wx = self.wlx[i][:,:Nkg]  ; wa = self.wla[i][:,:Ngg]
            Cggkg = np.concatenate((np.dot(wa,Cgg),np.dot(wx,Ckg)))
            # STACKING, MAKE THIS MORE GENERAL
            Nl,Nmon = Cggkg.shape
            res = np.zeros((Nl,1+(Nmon-1)*4))
            res[:,0] = Cggkg[:,0]
            res[:,1+i*(Nmon-1):4+i*(Nmon-1)] = Cggkg[:,1:]
            full_pred.append(res)
        return np.concatenate(full_pred)