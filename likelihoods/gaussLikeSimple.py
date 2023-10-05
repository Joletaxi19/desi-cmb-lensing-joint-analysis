import numpy as np
from math import isinf

class gaussLike():
   """
   A Gaussian likelihood class. 
   
   If no analytic marginalization is requred, then all 
   log-likelihoods are "true" likelihoods up to a 
   constant = -ln(P(dat)). When analytic marginalization
   over a set of templates is required, all log-likelihoods 
   correspond to the ("true" likelihood) x (the priors of 
   the template coefficients), up to the same constant.
   
   ...
   
   Attributes
   ----------
   D : int
      number of data points
   T : int, default=0
      number of templates for analytic marginalization
   dat : (D) ndarray
      data vector
   cov : (D,D) ndarray
      covariance matrix
   thy : function
      returns theory prediction, which is a ndarray
      with shape (D) when T = 0 or (D,1+T) when T > 0
   tmp_priors : None OR (T,2) ndarray, default=None 
      Gaussian priors for coefficients multiplying theory 
      templates. tmp_priors[:,0] are the means while
      tmp_priors[:,1] are the standard deviations.
      
   Methods
   -------
   templatePrior(tmp_prm)
      Computes prior for template coefficients
   rawLogLike(thy_args, tmp_prm=None)
      Computes the 'raw' log-likelihood
   anaHelp(thy_args)
      Helper function for margLogLike and maxLogLike
   margLogLike(thy_args)
      Computes the analytically-marginalized log-likelihood
   maxLogLike(thy_args)
      Computes the log-likelihood for the best-fit
      template coefficients
   """

   def __init__(self, dat, cov, tmp_priors=None):
      """
      Parameters
      ----------
      dat : (D) ndarray
         data vector
      cov : (D,D) ndarray
         covariance matrix
      thy : function
         returns theory prediction, which is a ndarray
         with shape (D) when T = 0 or (D,1+T) when T > 0
      tmp_priors : None OR (T,2) ndarray, default=None 
         Gaussian priors for coefficients multiplying theory 
         templates. tmp_priors[:,0] are the means while
         tmp_priors[:,1] are the standard deviations.
      """
      
      self.dat        = dat
      self.cinv       = np.linalg.inv(cov)
      self.logdet     = np.sum(np.log(np.linalg.eigvalsh(2*np.pi*cov))) 
      self.tmp_priors = tmp_priors
      self.D          = len(dat)
      self.T          = 0
      if tmp_priors is not None:
         self.T       = tmp_priors.shape[0]
      
      
   def templatePrior(self,tmp_prm):
      """
      Computes prior for template coefficients
      
      Parameters
      ----------
      tmp_prm : (T) ndarray
         values of the T template coefficients
      """
      if self.T == 0.: return 1.
      delt   = np.array(tmp_prm) - self.tmp_priors[:,0]
      chi2   = np.sum((delt/self.tmp_priors[:,1])**2.)
      volfac = np.prod(2*np.pi*self.tmp_priors[:,1]**2)**0.5
      return np.exp(-0.5*chi2) / volfac
      
      
   def rawLogLike(self, thy, tmp_prm=None):
      """
      Computes the 'raw' log-likelihood
      
      Parameters
      ----------
      thy_args : dict
         inputs to self.thy function
      tmp_prm : None OR (T) ndarray, default=None
         values of the T template coefficients

      Raises
      ------
      RuntimeError
         If tmp_prm isn't specified but the number of 
         templates is > 0
      """
      full_thy = None
      
      if tmp_prm is None:
         if self.T != 0.: 
            s = "tmp_prm not specified, but the number"
            s += " of templates is > 0"
            raise RuntimeError(s)
         full_thy  = thy
      else: 
         monomials = np.array([1.]+list(tmp_prm))
         full_thy  = np.dot(thy, monomials)
      
      delt  = full_thy - self.dat
      chi2  = np.dot(delt,np.dot(self.cinv,delt))
      return -0.5*chi2 + np.log(self.templatePrior(tmp_prm))  - 0.5*self.logdet

    
   def anaHelp(self, thy):
      """
      Helper function for margLogLike and maxLogLike

      Parameters
      ----------
      thy_args : dict
         inputs to self.thy function

      Raises
      ------
      RuntimeError
         If this function is called when there
         are no templates (T=0)
      """
      if self.T == 0.: 
         s = "anaHelp should never need to be called"
         s += " when there are no templates"
         raise RuntimeError(s)
       
      A = thy[:,0]
      B = thy[:,1:]
      delt = A - self.dat
      CphiInv = np.diag(self.tmp_priors[:,1]**-2.)
      Minv = CphiInv + np.matmul(B.T,np.matmul(self.cinv,B))
      M = np.linalg.inv(Minv)
      V = [np.dot(B[:,i],np.dot(self.cinv,delt)) for i in range(self.T)]
      V = np.array(V) - np.dot(CphiInv,self.tmp_priors[:,0])
      return delt,M,V
      
      
   def margLogLike(self, thy):
      """
      Computes the analytically-marginalized log-likelihood
      
      Parameters
      ----------
      thy_args : dict
         inputs to self.thy function
      """
      if self.T == 0.: 
         return self.rawLogLike()
         
      delt,M,V = self.anaHelp(thy)
      
      prefac  = self.templatePrior(np.zeros(self.T))
      chi2    = np.dot(delt,np.dot(self.cinv,delt))
      chi2    = chi2-np.dot(V,np.dot(M,V))
      try:
        logdetM = 0.5*np.log(np.linalg.det(2*np.pi*M))
      except:
        print('Overflow, trying [log det(M) = sum log eigvals] instead')
        eigvals = np.linalg.eigvalsh(2*np.pi*M)
        if np.any(eigvals<0):
            print('Found negative eigenvalues')
            return np.nan
        logdetM = 0.5*np.sum(np.log(eigvals))
      res = np.log(prefac) - 0.5*chi2 - 0.5*self.logdet + logdetM

      return res
   
      
   def maxLogLike(self, thy):
      """
      Computes the log-likelihood for the best-fit
      template coefficients
      
      Parameters
      ----------
      thy_args : dict
         inputs to self.thy function
      """
      if self.T == 0:
         return self.rawLogLike(thy)
          
      delt,M,V = self.anaHelp(thy)
      tmp_prm_star = -1*np.dot(M,V)
      return self.rawLogLike(thy,tmp_prm=tmp_prm_star)

   def getBestFitTemp(self, thy):
      """
      Computes the best-fit values
      of the template parameters.
      """
      delt,M,V = self.anaHelp(thy)
      tmp_prm_star = -1*np.dot(M,V)
      return tmp_prm_star

   def getBestFit(self, thy):
      """
      Returns theory prediction when all
      of the linear parameters are fixed
      to their best-fit values.
      """
      tmp_prm_star = self.getBestFitTemp(thy)
      monomials = np.array([1.]+list(tmp_prm_star))
      return np.dot(thy,monomials)
