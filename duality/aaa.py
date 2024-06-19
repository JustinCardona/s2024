import numpy as np
import pickle
from scipy.linalg import eig

def bsxfun(func, x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    M = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            M[i, j] = func(x[i], y[j])
    return M

class pade:
    def __init__(self, function, Z, tol=1e-13, mmax=150):
        self.function = function
        self.tol = tol
        self.mmax = mmax
        if type(Z) == str:
            self.load(Z)
            return None
        
        self.Z = Z
        self.F = self.function(self.Z)
        R = np.max(np.mean(self.F))
        self.z = np.empty(0)
        self.f = np.empty(0)
        for m in range(min(self.mmax, Z.shape[0]-1)):
            j = np.argmax(self.F-R)

            self.z = np.append(self.z, self.Z[j])
            self.f = np.append(self.f, self.F[j])
            self.Z = np.delete(self.Z, j)
            self.F = np.delete(self.F, j)
            
            A = self.loewner()
            C = self.cauchy()

            _, _, V = np.linalg.svd(A)
            self.w = V[m]
            N = C @ np.multiply(self.w, self.f)
            D = C @ self.w        
            R = np.divide(N, D)
            if np.linalg.norm(self.F-R) <= self.tol * np.linalg.norm(self.F, np.inf):
                break


    def prz(self):
        m = self.w.shape[0]
        B = np.eye(m+1)
        B[0, 0] = 0
        E = np.vstack((np.append(np.array([0]), self.w), np.hstack((np.ones((m, 1)), np.diag(self.z)))))
        pol = eig(E, B)[0]
        pol = pol[np.isfinite(pol)]
        dz = 1e-5 * np.exp(2j * np.pi * np.arange(1, 4) / 4)
        res = bsxfun(lambda x, y: self.eval(x + y), pol, dz) @ dz / 4

        
        E = np.vstack((np.append(np.array([0]), np.multiply(self.w, self.f)), np.hstack((np.ones((m, 1)), np.diag(self.z)))))

        zer = eig(E, B)[0] 
        zer = zer[np.isfinite(zer)]
        return pol, res, zer
    

    def cleanup(self, tol=1e-13):
        m = self.z.shape[0]
        P, R, Z = self.prz()
        P = P[np.where(R < tol)]
        for p in P:
            idx = np.argmin(Z - p)
            self.z = np.delete(self.z, idx)
            self.f = np.delete(self.f, idx)
            R = np.delete(R, idx)
            Z = np.delete(Z, idx)
        self.fit()

    def fit(self, x=None):
        if x != None:
            self.Z = np.append(self.Z, x)
            self.F = np.append(self.F, self.function(x))

        A = self.loewner()
        try:
            _, _, V = np.linalg.svd(A)
            m = self.z.shape[0]
            self.w = V[m-1]
        except:
            self.z = self.z[:-1]
            self.f = self.f[:-1]
            print('This data point is confusing, and has been forbidden.')


    def eval(self, x):
        m = self.w.shape[0]
        n = sum(list(map(lambda j: self.w[j] * self.f[j] / (x - self.z[j]), range(m))))
        d = sum(list(map(lambda j: self.w[j] / (x - self.z[j]), range(m))))
        return n / d


    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump((self.w, self.z, self.Z, self.f, self.F), f)


    def load(self, file_name):
        with open(file_name, "rb") as f:
                self.w, self.z, self.Z, self.f, self.F = pickle.load(f)


    def loewner(self):
        m = self.z.shape[0]
        M = self.Z.shape[0]
        A = np.empty((M, m))
        for i in range(M):
            for j in range(m):
                A[i, j] = (self.F[i] - self.f[j]) / (self.Z[i] - self.z[j])
        return A
    

    def cauchy(self):
        m = self.z.shape[0]
        M = self.Z.shape[0]
        C = np.empty((M, m))
        for i in range(M):
            for j in range(m):
                C[i, j] = 1 / (self.Z[i] - self.z[j])
            return C
