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
        self.dt = dt
        self.t += dt
        self.iter += 1
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)
class ImplicitTimestepper(Timestepper):
    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)
class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        self.steps = steps
        self.svalue = 1
        self.memory = np.vstack([np.transpose(self.f(np.transpose(u))),])
    def _step(self, dt):
        self.dt
        pass
    
    @staticmethod
    def coeff(steps,deltas):
        k = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        b[1,1]=1
        rows,cols = np.shape(k)
        for i in range(rows):
            for j in range(cols):
                k[i,j] = (-1)**(i+1)*(deltas[j])**(i+1)/np.math.factorial(i+1)
        print(k)
        x = np.linalg.solve(k,b)
        return x
