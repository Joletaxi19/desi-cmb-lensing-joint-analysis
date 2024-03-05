import numpy as np
from   numpy.random import multivariate_normal as normal
from   scipy.stats  import chi2                as CHI2

pte = lambda chi2,dof: 1.-CHI2.cdf(chi2,df=dof)

class gaussLike():
   """
   A Gaussian likelihood class. 
   
   If no analytic marginalization is requred, then all 
   log-likelihoods are "raw" likelihoods up to a 
   constant. When analytic marginalization over a set of 
   templates (mutiplied by linear coefficients) is required, 
   all log-likelihoods correspond to the ("raw" likelihood) x 
   (the priors of the template coefficients), up to a constant.
   """
   def __init__(self, dat, cov, tmp_priors=None, jeffreys=False):
      """
      Parameters
      ----------
      dat : (D) ndarray
         data vector
      cov : (D,D) ndarray
         covariance matrix
      tmp_priors : None OR (T,2) ndarray, default=None 
         Theory prediction is a ndarray with shape (D,1+T). 
         The last T columns are the templates to be marginaled over.
         tmp_priors specifices the Gaussian priors on the coefficients 
         multiplying those templates. tmp_priors[:,0] are the means 
         while tmp_priors[:,1] are the standard deviations.
      jeffreys: bool, default=False
         If True, include a partial Jeffrey's prior on the linear
         parameters.
      """
      
      self.dat        = dat
      self.cinv       = np.linalg.inv(cov)
      self.tmp_priors = tmp_priors
      self.D          = len(dat)
      self.T          = 0
      self.jeff       = jeffreys
      if tmp_priors is not None:
         self.T       = tmp_priors.shape[0]

   def templatePrior(self,tmp_prm):
      """
      Computes (log-)prior for template coefficients
      up to a constant.
      
      Parameters
      ----------
      tmp_prm : (T) ndarray
         values of the T template coefficients
      """
      if self.T == 0.: return 1.
      delt   = np.array(tmp_prm) - self.tmp_priors[:,0]
      chi2   = np.sum((delt/self.tmp_priors[:,1])**2.)
      return -0.5*chi2 

   def rawLogLike(self, thy, tmp_prm=None):
      """
      Computes the 'raw' log-likelihood
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
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
      
      delt = full_thy - self.dat
      chi2 = np.dot(delt,np.dot(self.cinv,delt))
      res  = -0.5*chi2 + self.templatePrior(tmp_prm)
      if self.jeff:
         _,M,_ = self.anaHelp(thy)
         res  -= 0.5*np.log(np.linalg.det(M))
      return res

   def anaHelp(self, thy):
      """
      Helper function for margLogLike and maxLogLike

      Parameters
      ----------
      thy: ndarray
         theory prediction tables

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
      thy: ndarray
         theory prediction tables
      """
      if self.T == 0.: 
         return self.rawLogLike()
         
      delt,M,V = self.anaHelp(thy)
      chi2     = np.dot(delt,np.dot(self.cinv,delt))
      chi2     = chi2-np.dot(V,np.dot(M,V))
      try:
        logdetM = 0.5*np.log(np.linalg.det(M))
      except:
        print('Overflow, trying [log det(M) = sum log eigvals] instead')
        eigvals = np.linalg.eigvalsh(M)
        if np.any(eigvals<0):
            print('Found negative eigenvalues')
            return np.nan
        logdetM = 0.5*np.sum(np.log(eigvals))
      res = -0.5*chi2 + logdetM*(not self.jeff)
      return res
   
   def getBestFitTemp(self, thy):
      """
      Computes the best-fit values
      of the template (linear) parameters.
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      """
      delt,M,V = self.anaHelp(thy)
      tmp_prm_star = -1*np.dot(M,V)
      return tmp_prm_star

   def maxLogLike(self, thy):
      """
      Computes the log-likelihood for the best-fit
      template coefficients
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      """
      if self.T == 0:
         return self.rawLogLike(thy)    
      tmp_prm_star = self.getBestFitTemp(thy)
      return self.rawLogLike(thy,tmp_prm=tmp_prm_star)

   def getBestFit(self, thy):
      """
      Returns theory prediction when all
      of the linear parameters are fixed
      to their best-fit values.
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      """
      tmp_prm_star = self.getBestFitTemp(thy)
      monomials = np.array([1.]+list(tmp_prm_star))
      return np.dot(thy,monomials)

   def margchi2(self, thy):
      """
      Returns the (data) chi2 averaged over linear parameters (analytically).
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      """
      A = thy[:,0]
      B = thy[:,1:]
      delt,M,V = self.anaHelp(thy)
      X = np.array([np.dot(B[:,i],np.dot(self.cinv,delt)) for i in range(self.T)]) 
      Y = np.dot(M,V)
      Z = np.matmul(B.T,np.matmul(self.cinv,B))  
      W = M + np.outer(Y,Y)  
      chi2 = np.dot(np.dot(delt,self.cinv),delt)
      chi2-= 2*np.dot(X,Y)
      chi2+= np.trace(np.matmul(Z,W))
      return chi2
    
   def bfchi2(self, thy):
      """
      Returns the (data) chi2 for the best-fit linear parameters.
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      """
      d = self.getBestFit(thy) - self.dat
      return np.dot(np.dot(d,self.cinv),d)
    
   def get_random_tmp_prm(self, thy, Ndraw):
      """
      Returns list (len = Ndraw) of linear ("template") paramaters 
      that are randomly drawn from the appropriate Gaussian
      distribution
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      Ndraw: int
         number of (Monte-Carlo) integration points
         used for the linear parameters
      """
      delt,M,V = self.anaHelp(thy)
      rescale  = np.diag(M)**0.5
      def get_random_draw():
         # rescaling the template parameters by their standard deviations
         # improves the convergence of the Monte-Carlo integration
         # (I think numpy isn't very good at making Gaussian draws when 
         #  there is a large dynamic range in the Gaussian variables)
         # I checked this by comparing the Monte-Carlo integration of the average chi2
         # with the analytic calculation (margchi2)
         return normal(-1.*np.dot(M,V)/rescale,M/np.outer(rescale,rescale))*rescale
      return [get_random_draw() for i in range(Ndraw)]
        
   def marg_chi2_pte(self, thy, Ndraw=100):
      """
      Returns the chi2 and PTE averaged over linear parameters (using MC integration).
      
      Parameters
      ----------
      thy: ndarray
         theory prediction tables
      Ndraw: int
         number of (Monte-Carlo) integration points
         used for the linear parameters
      """
      random_tmp_prms = self.get_random_tmp_prm(thy, Ndraw=Ndraw)
      res = []
      for tmp_prm in random_tmp_prms:
         delt = np.dot(thy,np.array([1.]+list(tmp_prm))) - self.dat
         chi2 = np.dot(np.dot(delt,self.cinv),delt)
         res.append([chi2,pte(chi2,self.D)])
      return np.mean(res,axis=0)