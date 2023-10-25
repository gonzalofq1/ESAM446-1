from timesteppers import StateVector
from scipy import sparse
import numpy as np

class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

      
        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
        f = lambda X: 0*X.data
        self.F = f

        
class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

   
        self.M = I
        L00 = -D*d2.matrix-I*c_target
        
        if (type(c_target) == np.ndarray):
            L00 = -D*d2.matrix-sparse.diags(c_target,0)

        self.L = L00

        f = lambda X: -X.data**2
        self.F = f



class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I*rho0 
        M01 = Z
        M10 = Z
        M11 = I
        
        if (type(rho0) == np.ndarray):
            M00 = sparse.diags(rho0,0)

        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        
        L00 = Z
        L01 =  d.matrix
        L10 =  gammap0*d.matrix
        L11 = Z

        if (type(gammap0) == np.ndarray):
            L10 = d.matrix @ sparse.diags(gammap0,0) 

        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        f = lambda X: 0*X.data
        self.F = f


