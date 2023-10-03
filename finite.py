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
        

        r = m+1-1+n
        p = (r-1)/2
        S = np.zeros((r,r))
        ps = np.linspace(-p,p,r)
        rows = np.linspace(0,2*p,r)
        h=(grid[-1]-grid[0])/grid.size
        shape  = np.shape(S)
        for i in range(shape[0]):
            for j in range(shape[1]):
                S[i,j]=(ps[j]*h)**i/np.math.factorial(i)
        self.stencil = S


class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        pass

