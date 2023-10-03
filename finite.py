import numpy as np
from scipy.special import factorial
from scipy import sparse

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Difference:

    def __matmul__(self, other):
        return self.matrix@other


class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        m = derivative_order
        n = convergence_order

        r = int((m+1)/2)*2-1+n
        p = (r-1)/2
        S = np.zeros((r,r))
        ps = np.linspace(-p,p,r)
        rows = np.linspace(0,2*p,r)
        h=grid.dx
        shape  = np.shape(S)
        for i in range(shape[0]):
            for j in range(shape[1]):
                S[i,j]=(ps[j]*h)**i/np.math.factorial(i)
        
        b = np.zeros((r,1))
        b[m]=1
 

        a = np.linalg.inv(S) @ b 
        
        offsets = ps.astype(int)
     
        shape = [int(grid.N),int(grid.N)]
        D = sparse.diags(a, offsets=offsets, shape=shape)
        D = D.tocsr()
        for j in range(int((r-1)/2)):
            for i in range(int((r-1)/2)-j):
                D[j,i-int((r-1)/2)+j]=a[i]
            
        for j in range((int((r-1)/2))):
            for i in range((int((r-1)/2))-j):
                D[-1-j,i]=a[int((r-1)/2)+1+i+j]
        
        self.matrix = D
        print(np.shape(D))



class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        pass
