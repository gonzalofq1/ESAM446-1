from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import finite
import numpy as np
import timesteppers
import scipy.sparse.linalg as spla
import timesteppers


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

        
        self.F = lambda X: 0*X.data


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


class ReactionDiffusion2D: 
    
    def __init__(self, c, D, dx2,dy2):
        self.t = 0 
        self.iter = 0 
        self.V = timesteppers.StateVector([c])
        diffx = self.Diffusionx(c,D, dx2)
        diffy = self.Diffusiony(c, D, dy2)
        reaction = self.Reaction(c)
        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)
        self.ts_r = timesteppers.RK22(reaction)
        

    
    class Diffusionx:

        def __init__(self, c, D, dx2):
            self.X = timesteppers.StateVector([c], axis=0)
            N = c.shape[0]
            self.M = sparse.eye(N, N)
            self.L = -D*dx2.matrix


    class Diffusiony:

        def __init__(self, c, D, dy2):
            self.X = timesteppers.StateVector([c], axis=1)
            N = c.shape[1]
            self.M = sparse.eye(N, N)
            self.L = -D*dy2.matrix
            
    class Reaction:

        def __init__(self, c):
            self.X = timesteppers.StateVector([c])
            N = c.shape[0]
            self.M = sparse.eye(N, N)
            self.L = sparse.csr_matrix((N, N))
            f = lambda X: X.data-X.data**2
            self.F = f
            
    

    def step(self,dt):
        self.t = self.t+dt
        self.iter = self.iter+1
        
        self.ts_r.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_r.step(dt/2)
        
class ViscousBurgers2D:

    def __init__(self, u, v, nu, order, domain):
        self.X = StateVector([u, v])
        self.t = 0
        self.iter = 0


        x = domain.grids[0]
        y = domain.grids[1]


        dx2 = finite.DifferenceUniformGrid(2, order, x, 0)
        dy2 = finite.DifferenceUniformGrid(2, order, y, 1)
        dx = finite.DifferenceUniformGrid(1, order, x, 0)
        dy = finite.DifferenceUniformGrid(1, order, y, 1)

        diffx = self.Diffusion(u,v,nu, dx2,0)
        diffy = self.Diffusion(u,v, nu, dy2,1)
        adv = self.Advection(u,v,dx,dy)


        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)
        self.ts_adv = timesteppers.RK22(adv)


    class Diffusion:

        def __init__(self, u,v, nu, d2,axis):
            self.X = timesteppers.StateVector([u,v], axis)
            N = len(u)
            self.M = sparse.eye(N, N)
            I = sparse.eye(N, N)
            Z = sparse.csr_matrix((N, N))
            L = -nu*d2.matrix
            M00 = I
            M01 = Z
            M10 = Z
            M11 = I
            self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])

            L00 = L
            L01 = Z
            L10 = Z
            L11 = L
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])




    class Advection:

        def __init__(self, u,v, dx,dy):
            self.X = timesteppers.StateVector([u,v])

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
            L01 = Z
            L10 = Z
            L11 = Z
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])


            def f(X):
                u1 = np.copy(X.data[:N,:])
                v1 = np.copy(X.data[N:,:])
                w = -np.vstack((v1*(dy@u1)+u1*(dx@u1),u1*(dx@v1)+v1*(dy@v1)))
                return w

            self.F =f 


    def step(self,dt):
        self.t = self.t+dt
        self.iter = self.iter+1


        self.ts_adv.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_adv.step(dt/2)
class DiffusionBC:

    def __init__(self, c, D, spatial_order, domain):
        self.c = c
        self.X = timesteppers.StateVector([c])
        self.t = 0
        self.iter = 0
       
        x = domain.grids[0]
        y= domain.grids[1]
                
        dx = finite.DifferenceUniformGrid(1, spatial_order, x)
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, x)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, y)


        diffx = self.Diffusionx(c,D, dx,dx2)
        diffy = self.Diffusiony(c, D, dy2)
        
        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)

    class Diffusionx:

        def __init__(self, c, D, dx,dx2):
            self.X = timesteppers.StateVector([c], axis=0)
            N = c.shape[0]
            M = sparse.eye(N, N)
            M=M.tocsr()
            M[0,0]=0
            M[-1,-1]=0
            M.eliminate_zeros()
            self.M = M

            L = -D*dx2.matrix
            L=L.tocsr()
            L[0,:] = 0
            L[-1,:] = 0
            L[0,0] = 1
            L[-1,:] = dx.matrix[-1,:]
            L.eliminate_zeros()
            self.L = L

    
    class Diffusiony:
    
        def __init__(self, c, D, dy2):
            self.X = timesteppers.StateVector([c], axis=1)
            N = c.shape[1]
            self.M = sparse.eye(N, N)
            self.L = -D*dy2.matrix
    
        def step(self, dt):
            self.t = self.t+dt
            self.iter = self.iter+1
        
            self.ts_y.step(dt/2)
            self.ts_x.step(dt/2)
            self.ts_x.step(dt/2)
            self.ts_y.step(dt/2)
    
    
class Wave2DBC:

    def __init__(self, u, v, p, spatial_order, domain):
        self.X = timesteppers.StateVector([u,v,p])
        N = len(u)
        x = domain.grids[0]
        y= domain.grids[1]     
    
        self.dx = finite.DifferenceUniformGrid(1, spatial_order, x)
        self.dy = finite.DifferenceUniformGrid(1, spatial_order, y,1)
    
        def BC(X):
            X.data[0,:]=0
            X.data[N-1,:]=0
        self.BC = BC       
        
    
        def f(X): 
            v1 = -(self.dx @ X.data[2*N:,:])
            v2 = -(self.dy @ X.data[2*N:,:])
            v3 = -(self.dx @ X.data[0:N,:])-(self.dy @ X.data[N:2*N,:])
            return np.vstack((v1,v2,v3))
        self.F = f
    
    
