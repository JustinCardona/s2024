import numpy as np
import pickle

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

    
    def fit(self, x):
        self.z = np.append(self.z, x)
        self.f = np.append(self.f, self.function(x))

        A = self.loewner()
        try:
            _, _, V = np.linalg.svd(A)
            m = self.z.shape[0]
            self.w = V[m-1]
        except:
            self.z = self.z[:-1]
            self.f = self.f[:-1]
            print("invalid data point")


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