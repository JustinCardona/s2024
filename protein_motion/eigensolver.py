import numpy as np
import time

def givensrotation(a, b):
    hypot = np.sqrt(a**2 + b**2)
    cos = a / hypot
    sin = -b / hypot
    return cos, sin


def qr_givens(A, width):
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for i in range(0, n - 1):
        end = min(m, i+width+1)
        for j in range(i + 1, end):
            cos, sin = givensrotation(R[i, i], R[j, i])
            R[i], R[j] = (R[i] * cos) + (R[j] * (-sin)), (R[i] * sin) + (R[j] * cos)
            Q[:, i], Q[:, j] = (Q[:, i] * cos) + (Q[:, j] * (-sin)), (Q[:, i] * sin) + (Q[:, j] * cos)
            # print(np.round(R, 2))
    return Q, R


# GENERATE REAL SYMMETRIC BANDED MATRIX
n = 10
m = 1
np.random.seed(0)
A = np.array(np.random.randint (0, 10, (n, n)), dtype=np.float64)
A = 0.5 * (A + A.T)
for i in range(n):
    for j in range(n):
        if np.abs(i-j) > m:
            A[i, j] = 0

# QR DECOMPOSITION
t = time.time()
Q, R = qr_givens(A, m)
print(time.time() - t)
print("Q:")
print(np.round(Q, 2))
print("R:")
print(np.round(R, 2))