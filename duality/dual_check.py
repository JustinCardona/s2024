from aaa import *
import matplotlib.pyplot as plt


def dual_check(func, xi_0, tol=1e-3, depth_max=500):
    xi_prev = xi_0
    xi_cur = 1.1 * xi_prev

    domain = np.linspace(-xi_0, xi_0, 300)
    a = pade(func, domain)
    err = np.linalg.norm(func(xi_cur))
    depth_cur = 0
    while err > tol and depth_cur < depth_max:
        xi_prev = xi_cur
        a.fit(xi_prev)
        _, _, z = a.prz()
        xi_cur = np.max(z)
        err = np.linalg.norm(func(xi_cur))
        depth_cur += 1
    return a, xi_cur, err
        

def rand(shape):
    M = np.empty(shape, dtype=complex)
    if len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                M[i, j] = np.random.rand() + np.random.rand()*1j
    else:
        for i in range(shape[0]):
            M[i] = np.random.rand() + np.random.rand()*1j
    return M

n = 10
a0 = rand([1])
a1 = rand([1])
s  = rand([n])
P0 = rand((n, n))
P1 = rand((n, n))
P1 = (P1.conj().T + P1) / 2
def surr(xi):
    M = (a1 + xi) * P0 + a0 * P1
    x = np.linalg.solve(M, s)
    return ((x.conjugate() @ s).imag - x.conjugate() @ P0 @ x).real
surr = np.vectorize(surr)


a, xi, err = dual_check(surr, 1)
Z = np.linspace(-xi, xi, 300)
print(np.format_float_scientific(xi.real, precision=2), "pm", np.format_float_scientific(err, precision=2))
plt.plot(Z, surr(Z))
plt.plot(Z, a.eval(Z))
plt.show()