import sys
import numpy as np
import scipy as sp
from ._generic_model import GenericModel

###############################################################################

def p_norm(x, p=2.0):
    """
    Parameters
    ----------
    x: array
        Vector to have the norm calculated;
        norm_p(x) = ( sum (|x_i|^p) ) ^(1/p)
    """
    return np.sum((np.abs(x)**p))**(1/p)

###############################################################################

def calculate_vector_similarity(x, y, p=2.0,functional_form='rbf',
                                gamma=1.0, eta=1e-10):
    """
    Parameters
    ----------        
    x, y: array
        Vectors to have the similarity calculated.
    p: float, optional
        p_value for evaluating the function p_norm(x,p_value)
    functional_form: string, optional
        rbf = radial basis function; 
        ies = inverse euclidean similarity; 
        iqk = inverse quadratic kernel; 
        exp_kernel = exponential kernel; 
        cauchy_kernel = cauchy kernel.
    gamma: float, optional
        Parameter used in the functional form.
    eta: float, optional
        Minimum similarity value to be returned.
        
    Returns
    ----------        
    : float
        Similarity.
    """
    dif = x-y
    
    if functional_form=='rbf':
        sim = np.exp(-gamma*p_norm(dif,p)**2)
               
    elif functional_form=='ies':
        sim = 1./(1+gamma*p_norm(dif,p))
        
    elif functional_form=='iqk':
        sim = 1./(np.sqrt(1+(gamma*p_norm(dif,p))**2))

    elif functional_form=='exp_kernel':
        sim = np.exp(-gamma*p_norm(dif,p))
        
    elif functional_form=='cauchy_kernel':
        sim = 1./(1+(gamma*p_norm(dif,p))**2)

    if sim == 0.: sim = eta

    return sim

###############################################################################

def calculate_matrix_similarity(A, B, p=2.0,functional_form='rbf',
                                gamma=1.0, eta=1e-10):
    
    """
    Parameters
    ----------        
    A: numpy.array
        Left matrix (vectors in row).
    B: numpy.array
        Right matrix (vectors in column).
    p: float, optional
        p_value for evaluating the function p_norm(x,p_value)
    functional_form: string, optional
        rbf = radial basis function; 
        ies = inverse euclidean similarity; 
        iqk = inverse quadratic kernel; 
        exp_kernel = exponential kernel; 
        cauchy_kernel = cauchy kernel.
    gamma: float, optional
        Parameter used in the functional form.
    eta: float, optional
        Minimum similarity value to be returned.
        
    Returns
    ----------        
    : numpy.array
        Similarity matrix.
    """
    if A.ndim < 2:
        A = np.reshape(A, (len(A), -1))
    if B.ndim < 2:
        B = np.reshape(B, (len(B), -1))
    
    rows = np.shape(A)[0]
    columns = np.shape(B)[1]
    C = np.zeros((rows,columns))
        
    for i in range(rows):
        for j in range(columns):
            C[i,j]=calculate_vector_similarity(A[i,:],B[:,j], 
                                               p,
                                               functional_form,gamma,eta)
    
    return C

###############################################################################

class SBM (GenericModel):
    """
    Similarity-based method (SBM).
    
    For details on the technique, see the papers:
        
        * Marins et al. (2018) - Improved similarity-based modeling for the 
          classification of rotating-machine failures, 
          http://www02.smt.ufrj.br/~sergioln/papers/IJ28.pdf
        * Ribeiro (2018) - Similarity-based methods for machine diagnosis, 
        http://www.pee.ufrj.br/index.php/pt/producao-academica/teses-de-doutorado/tese-1/2016033299-similarity-based-methods-for-machine-diagnosis/file
    
    Parameters
    ----------  
        p: float, optional
            p-value for the definition of the norm.
        functional_form: string, optional
            Functional form to be used in the similarity calculation.
            rbf = radial basis function; 
            ies = inverse euclidean similarity; 
            iqk = inverse quadratic kernel; 
            exp_kernel = exponential kernel; 
            cauchy_kernel = cauchy kernel.
        gamma: float, optional
            Parameter present in the various functional forms 
            of similarity.
        eta: float, optional
            Minimum value to be returned in similarity calculations.
        train_method: string, optional
            Training method. Options: 'all_archetypes' and
            'geometrical_median'.
        tau: float, optional
            Similarity threshold.
        verbose: boolean, optional
            Whether to print information during execution.
    """    

    ###########################################################################

    def __init__ (self, p = 2.0, functional_form = 'rbf', gamma = 1.0, 
                  eta = 1e-10, train_method = 'geometrical_median',
                  tau = 1.e-10, verbose = False):

        self.has_Y = False

        self.p = p
        self.function = functional_form
        self.gamma = gamma
        self.eta=eta
        self.train_method = train_method
        self.tau = tau
        self.verbose = verbose
        
        self.name = 'SBM'
        
    ###########################################################################
    
    def train_core (self):
     
        M = self.X_train.values
        if self.verbose:
            print('#########################################')
            print('Training SMB')
            print('')
            print('Full memory matrix with ',np.shape(M)[0],
                  ' archetypal states')
            print('')
            print('applying the ' , self.train_method, 
                  'method of training...' )
            print('')
        
        if self.train_method=='all_archetypes':
            self.D = M
        #elif method == 'original':
        
        elif self.train_method == 'geometrical_median':
            
            mean = np.mean(M,axis=0)
            
            minor = sys.maxsize
            for i in range(np.shape(M)[0]):
                dist_i = p_norm((M[i,:]-mean), p=2.0)
                if dist_i<minor:
                    minor = dist_i
                    median = M[i,:]
            
            self.D = median
            for i in range(np.shape(M)[0]):
                x = M[i,:].T
                C = calculate_matrix_similarity(self.D, x,self.p,
                                                self.function,
                                                self.gamma,self.eta)
                sim_mean = np.mean(C)
                if sim_mean<self.tau:
                    self.D=np.vstack((self.D, x.T))
                    
        self.Dt = self.D.T
        self.invsimDDt = sp.linalg.inv(calculate_matrix_similarity(self.D, 
                                                                   self.Dt,
                                                                   self.p,
                                                                 self.function,
                                                                   self.gamma,
                                                                   self.eta))

        if self.verbose:

            print('SBM trained - D matrix with ', 
                  np.shape(self.D)[0],' archetypal states')
            print('')
            print('#########################################')
        
        
    #########################
   
    def map_from_X(self, X):
        
        Xt = X.T
        n = np.shape(X)[0] # quantity of current states
        
        X_est = np.zeros(np.shape(X))
        
        for i in range(n):
            w = self.invsimDDt@calculate_matrix_similarity(self.D, Xt[:,i],
                                           self.p,self.function,
                                           self.gamma,self.eta)
            normalizer = np.sum(w)
            w = w/normalizer
            
            x = self.Dt@w 
            X_est[i,:]= x.T
            
        return X_est
        
