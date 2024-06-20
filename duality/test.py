import numpy as np

n = 5
z = np.random.rand(n)
Z = np.random.rand(n)
f = np.random.rand(n)
F = np.random.rand(n)

def loewner():
    m = z.shape[0]
    M = Z.shape[0]
    A = np.empty((M, m))
    for i in range(M):
        for j in range(m):
            A[i, j] = (F[i] - f[j]) / (Z[i] - z[j])
    return A


def cauchy():
    m = z.shape[0]
    M = Z.shape[0]
    A = np.empty((M, m))
    for i in range(M):
        for j in range(m):
            A[i, j] = 1 / (Z[i] - z[j])
    return A
    

A = loewner()
C = cauchy()
SF = np.diag(F)
Sf = np.diag(f)
M = SF @ C - C @ Sf

print(A, '\n')
print(C, '\n')
print(SF, '\n')
print(Sf, '\n')
print(A - M, '\n')