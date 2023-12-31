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

class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J
    
class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        self.X = timesteppers.StateVector([u])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        d1 = finite.DifferenceUniformGrid(1, spatial_order, grid)
        self.N = len(u)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -nu*d2.matrix

        def F(X):
            return -X.data*(d1 @ X.data)
        self.F = F
        
        def J(X):
            u_matrix = -sparse.diags(d1 @ X.data)
            dev = -sparse.diags(X.data) @ (d1.matrix)
            return u_matrix + dev
        
        self.J = J


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X

        N = len(self.X.variables[0])
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                                [M10, M11]])

        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        L00 = -D*d2.matrix
        L01 = Z
        L10 = Z
        L11 = -D*d2.matrix
        self.L = sparse.bmat([[L00, L01],
                                [L10, L11]])
        
        def F(X):
            c1 = self.X.data[0:N]
            c2 = self.X.data[N:]
            return np.append([c1*(1-c1-c2)],[r*c2*(c1-c2)])
        self.F = F


        def J(X):
            c1 = sparse.diags(self.X.data[0:N])
            c2 = sparse.diags(self.X.data[N:])
            m1 = sparse.eye(N)-2*c1-c2
            m2 = -c1
            m3 = r*c2
            m4 = r*c1-2*c2*r
            mat = sparse.bmat([[m1, m2],
                                [m3, m4]])
            return mat
        
        self.J = J
    
class Schrodinger2D:

    def __init__(self, u, v, order, domain):
        self.X = StateVector([u, v])
        self.t = 0
        self.iter = 0


        x = domain.grids[0]
        y = domain.grids[1]


        dx2 = finite.DifferenceUniformGrid(2, order, x, 0)
        dy2 = finite.DifferenceUniformGrid(2, order, y, 1)

        diffx = self.Diffusion(u,v, dx2,0)
        diffy = self.Diffusion(u,v, dy2,1)
        adv = self.Advection(u,v,x,y)


        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)
        self.ts_adv = timesteppers.RK22(adv)
        N = len(u)
        

          

    class Diffusion:

        def __init__(self, u,v, d2,axis):
            self.X = timesteppers.StateVector([u,v], axis)
            N = len(u)
            self.M = sparse.eye(N, N,dtype=complex)
            I = sparse.eye(N, N,dtype=complex)
            Z = sparse.csr_matrix((N, N),dtype=complex)
            L = d2.matrix.astype(complex)
            M00 = I
            M01 = Z
            M10 = Z
            M11 = I
            self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])

            L00 = L
            L01 = Z
            L10 = Z
            L11 = -L
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])




    class Advection:

        def __init__(self, u,v,x,y):
            self.X = timesteppers.StateVector([u,v])

            N = len(u)
            I = sparse.eye(N, N,dtype=complex)
            Z = sparse.csr_matrix((N, N),dtype=complex)

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
                w = np.vstack((v1*(u1**2+v1**2),-u1*(u1**2+v1**2)))
                return w*0

            self.F =f

            def BC(X):
                dx = x.dx
                dy = y.dx
                u1 = np.copy(X.data[:N,:])
                v1 = np.copy(X.data[N:,:])
                constant = np.sum(dx*dy*u1**2)+np.sum(dx*dy*v1**2)
                X.data[:,:]=X.data[:,:]/constant
                #print(constant)
            self.BC =BC

    
    def step(self,dt):
        self.t = self.t+dt
        self.iter = self.iter+1


        self.ts_adv.step(dt/3)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_adv.step(dt/3)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_adv.step(dt/3)
        

        
class SchrodingerBCNonLinear:

    def __init__(self, c, spatial_order, domain,g):
        self.c = c
        self.X = timesteppers.StateVector([c])
        self.t = 0
        self.iter = 0
       
        x = domain.grids[0]
        y= domain.grids[1]
                
        dx = finite.DifferenceUniformGrid(1, spatial_order, x)
        dy = finite.DifferenceUniformGrid(1, spatial_order, y)
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, x)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, y)


        diffx = self.Diffusion(c, dx,dx2,0)
        diffy = self.Diffusion(c,dy,dy2,1)
        adv = self.Advection(c,x,y,g)
        
        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)
        self.ts_adv = timesteppers.RK22(adv)

    class Diffusion:

        def __init__(self, c, dx,dx2,axis):
            self.X = timesteppers.StateVector([c], axis)
            N = c.shape[axis]
            M = sparse.eye(N, N,dtype=complex)
            M=M.tocsr()
            M[0,0]=0
            M[-1,-1]=0
            M.eliminate_zeros()
            self.M = M

            L = dx2.matrix.astype(complex)
            L=L.tocsr()
            L = L*(-1j)/2
            L[0,:] = 0
            L[-1,:] = 0
            L[0,0] = 1
            L[-1,-1] = 1
            L.eliminate_zeros()
            self.L = L

    
    class Advection:

        def __init__(self,c,x,y,g):
            self.X = timesteppers.StateVector([c])

            N = len(c)
            I = sparse.eye(N, N,dtype=complex)
            Z = sparse.csr_matrix((N, N),dtype=complex)

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
                adv = X.data*abs(X.data)**2
                return -1j*adv*g

            self.F =f

            def BC(X):
                X.data[0,:]=0
                X.data[-1,:]=0
                X.data[:,-1]=0
                X.data[:,0]=0
            self.BC =BC

    
    def step(self, dt):
        self.t = self.t+dt
        self.iter = self.iter+1
    
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_adv.step(dt/2)
        self.ts_adv.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        

        
class SchrodingerBCLinearSlit:

    def __init__(self, c, spatial_order, domain):
        self.c = c
        self.X = timesteppers.StateVector([c])
        self.t = 0
        self.iter = 0
       
        x = domain.grids[0]
        y= domain.grids[1]
                
        dx = finite.DifferenceUniformGrid(1, spatial_order, x)
        dy = finite.DifferenceUniformGrid(1, spatial_order, y)
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, x)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, y)


        diffx = self.Diffusion(c, dx,dx2,0)
        diffy = self.Diffusion(c,dy,dy2,1)
        adv = self.Advection(c)
        
        self.ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.ts_y = timesteppers.CrankNicolson(diffy, 1)
        self.ts_adv = timesteppers.RK22(adv)

    class Diffusion:

        def __init__(self, c, dx,dx2,axis):
            self.X = timesteppers.StateVector([c], axis)
            N = c.shape[axis]
            M = sparse.eye(N, N,dtype=complex)
            M=M.tocsr()
            M[0,0]=0
            M[-1,-1]=0
            M.eliminate_zeros()
            self.M = M

            L = dx2.matrix.astype(complex)
            L=L.tocsr()
            L = L*(-1j)/2
            L[0,:] = 0
            L[-1,:] = 0
            L[0,0] = 1
            L[-1,-1] = 1
            L.eliminate_zeros()
            self.L = L
            
    class Advection:

        def __init__(self,c):
            self.X = timesteppers.StateVector([c])

            N = len(c)
            I = sparse.eye(N, N,dtype=complex)
            Z = sparse.csr_matrix((N, N),dtype=complex)

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
                return 0

            self.F =f

            def BC(X):
                m = int(N/2)
                X.data[0:m-5,m-10:m+10]=0
                X.data[m+5:-1,m-10:m+10]=0
            self.BC =BC

    
    def step(self, dt):
        self.t = self.t+dt
        self.iter = self.iter+1
    
       
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_adv.step(dt/2)
        self.ts_adv.step(dt/2)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
           
