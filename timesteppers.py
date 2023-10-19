import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        super().__init__()
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a 
        self.b = b
        
       
        

    def _step(self, dt):
        k = np.zeros((len(self.u),self.stages))
        for i in range(self.stages):
            mult = np.transpose(self.a[i,:])
            k[:,i]=self.f(self.u+dt*np.matmul(k,mult))
        mult = np.transpose(self.b)
        return (self.u+np.matmul(k,self.b)*dt)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.dt = dt
        self.steps = steps
        self.svalue = 1
        self.memory = np.vstack([np.transpose(self.f(np.transpose(u))),])
        

    def _step(self, dt):
        if (self.svalue<self.steps):
            coff = self.coeff(self.svalue)
            rows = np.shape(self.memory)[0]
        
            next =np.matmul(np.transpose(coff),self.memory) 
            next = dt*next+self.u
            self.memory = np.vstack([np.transpose(self.f(np.transpose(next))),self.memory])
            self.svalue = self.svalue+1
            return next
        else:
           
            coff = self.coeff(self.steps)
            rows = np.shape(self.memory)[0]
            next =np.matmul(np.transpose(coff),self.memory) 
            next = dt*next+self.u
            self.memory = np.vstack([np.transpose(self.f(np.transpose(next))),self.memory[0:-1,:]])
            return next


    @staticmethod
    def coeff(steps):
        k = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        rows,cols = np.shape(k)
        for i in range(rows):
            for j in range(cols):
                k[i,j] = (-1)**(i)*(j)**(i)/np.math.factorial(i)
            b[i,0] = 1/np.math.factorial(i+1)
        x = np.linalg.solve(k,b)
        return x


class ImplicitTimestepper(Timestepper):
    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)
        
class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        super().__init__(u, L)
        self.steps = steps
        self.svalue = 1
    def _step(self, dt):
        if (self.svalue == 1):
            self.deltas = np.vstack([dt,])
            self.memory = np.vstack([np.transpose(self.u),])
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            new = self.LU.solve(self.u)
            self.svalue = self.svalue+1
            return new.flatten()

        elif (self.svalue<=self.steps):

            self.deltas = self.deltas+dt
            self.deltas = np.vstack([dt,self.deltas])
            self.memory = np.vstack([np.transpose(self.u),self.memory])
            coffs = self.coeffs(self.svalue,self.deltas)
            self.LHS =  self.I*np.sum(coffs) + (self.L.matrix)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')

            U = np.matmul(np.transpose(coffs),self.memory) 
            new = self.LU.solve(np.transpose(U))
            self.svalue = self.svalue+1
            return new.flatten()
        
        
        else:
            self.deltas = self.deltas+dt
            self.deltas = np.vstack([dt,self.deltas[0:-1,:]])
            self.memory = np.vstack([np.transpose(self.u),self.memory[0:-1,:]])
            coffs = self.coeffs(self.steps,self.deltas)
            self.LHS = self.I*np.sum(coffs) + (self.L.matrix)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            U = np.matmul(np.transpose(coffs),self.memory) 

            new = self.LU.solve(np.transpose(U))
            return new.flatten()
        
    
    @staticmethod
    def coeffs(steps,deltas):

        k = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        b[0,0]=1
        rows,cols = np.shape(k)
        for i in range(rows):
            for j in range(cols):
                k[i,j] = (-1)**(i+1)*(deltas[j,0])**(i+1)/np.math.factorial(i+1)
        x = np.linalg.solve(k,b)
        return x