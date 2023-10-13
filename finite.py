import numpy as np
from scipy.special import factorial
from scipy import sparse
#
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
                D[-1-j,i]=a[int((r-1)/2)+1+i+j]
        
        self.matrix = D
        print(a)



class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        
        gr = grid.values
        m = derivative_order
        n = convergence_order
        r = n+m
        p = int((r-1)/2)
        
        S = np.zeros((r,r))
        rows = np.linspace(0,2*p,r)
        ps = np.linspace(-p,p,r)
        shape  = np.shape(S)
        b = np.zeros((r,1))
        b[m]=1
        dev = np.zeros((gr.size,gr.size))

        
        for current in range(gr.size):
            dist = np.zeros(r)
            
            for i,w in enumerate(ps.astype(int)):
                if current+w<0:
                    dist[i] = -gr[current]+gr[(current+w)]-grid.length
                    
                elif current+w>=grid.N:
                    cur = (current+w)%(grid.N)
                    dist[i] = -gr[current]+gr[cur]+grid.length
                else:
                    dist[i] = -gr[current]+gr[(current+w)]
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    S[i,j]=(dist[j])**i/np.math.factorial(i)

            a = np.linalg.inv(S) @ b 
            for i in range(ps.size):
                if current+int(ps[i])>=grid.N:
                    dev[current,(current+int(ps[i]))%grid.N] = a[i]
                else:
                    dev[current,current+int(ps[i])] = a[i]
        
        self.matrix = dev

class ForwardFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix


