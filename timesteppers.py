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
    
class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.svalue = 1

    def _step(self, dt):
        if self.svalue == 1:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX

            self.memoryFX = np.copy(self.FX)
            self.memory = np.copy(self.X.data)
            self.deltas = np.vstack([dt,])
            self.svalue = self.svalue+1
            
            return LU.solve(RHS)
        
        elif (self.svalue<=self.steps):

    
            self.FX = self.F(self.X)
            self.memoryFX = np.column_stack((self.FX,self.memoryFX))
            self.memory = np.column_stack((self.X.data,self.memory))
            self.deltas = self.deltas+dt
            self.deltas = np.vstack([dt,self.deltas])
            coffsEXT = self.coeffEXT(self.svalue)
            coffsBDF = self.coeffBDF(self.svalue,self.deltas)
            F =  self.memoryFX @ coffsEXT
            U =  self.memory @ coffsBDF
        
            
            self.LHS = -self.M*np.sum(coffsBDF) + (self.L)
            self.RHS = -self.M @ U + F
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            self.svalue = self.svalue+1
            print(coffsEXT)
            return self.LU.solve(self.RHS).flatten()
        
        else:
            self.FX = self.F(self.X)
            self.deltas = self.deltas+dt
            self.deltas = np.vstack([dt,self.deltas[0:-1,:]])
            self.memory = np.column_stack((self.X.data,self.memory[:,0:-1]))
            self.memoryFX = np.column_stack((self.FX,self.memoryFX[:,0:-1]))
            coffsEXT = self.coeffEXT(self.steps)
            coffsBDF = self.coeffBDF(self.steps,self.deltas)
            F =  self.memoryFX @ coffsEXT
            U =  self.memory @ coffsBDF
            self.LHS = -self.M*np.sum(coffsBDF) + (self.L)
            self.RHS = -self.M @ U + F
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            return self.LU.solve(self.RHS).flatten()




    @staticmethod
    def coeffEXT(steps):

        k = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        b[0,0]=1
        rows,cols = np.shape(k)
        for i in range(rows):
            for j in range(cols):
                k[i,j] = (-1)**(i)*(j+1)**(i)/np.math.factorial(i)
        x = np.linalg.solve(k,b)
        return x
    
    @staticmethod
    def coeffBDF(steps,deltas):

        k = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        b[0,0]=1
        rows,cols = np.shape(k)
        for i in range(rows):
            for j in range(cols):
                k[i,j] = (-1)**(i+1)*(deltas[j,0])**(i+1)/np.math.factorial(i+1)
        x = np.linalg.solve(k,b)
        return x

