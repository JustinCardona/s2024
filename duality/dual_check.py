from aaa import *
import matplotlib.pyplot as plt

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


def find_root(func, a, b, tol=1e-7):
    c = (a + b) / 2
    while abs(func(c)) > tol:
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2 
    return c


class Surrogate:
    def __init__(self, n):
        self.n = n
        self.a0 = rand([1])
        self.a1 = rand([1])
        self.s  = rand([n])
        self.P0 = rand((n, n))
        self.P1 = rand((n, n))
        self.P1 = (self.P1.conj().T + self.P1) / 2

    def eval(self, xi):
        M = (self.a1 + xi) * self.P0 + self.a0 * self.P1
        x = np.linalg.solve(M, self.s)
        return ((x.conjugate() @ self.s).imag - x.conjugate() @ self.P0 @ x).real
    

def update_surr(s):
    dx = 1e-5
    s.P0 += dx * rand((s.n, s.n))
    s.P1 += dx * rand((s.n, s.n))


def dual_check(func_acc, surr_cur, surr_update_fn, xi_guess, depth, tol=1e-5):
    if depth == 0:
        return xi_guess, func_acc
    def approx(x):
        sum = 0
        for f in func_acc:
            sum += f.eval(x)
        return sum
    # surr = np.vectorize(surr_cur.eval)
    a = pade(lambda x: surr(x) - approx(x), np.linspace(0.9 * xi_guess, 1.1 * xi_guess, 3), tol, 2)
    _, _, z = a.prz()
    xi_guess = z[-1]
    func_acc.append(a)
    surr_update_fn(surr_cur)
    return dual_check(func_acc, surr_cur, surr_update_fn, xi_guess, depth-1)


xi_init = 1
n = 500
tol = 1e-11
domain = np.linspace(0.8 * xi_init, 1.2 * xi_init, n)
s_test = Surrogate(10)
surr = np.vectorize(s_test.eval)
pade_init = pade(surr, domain, tol, n//2)


Z = np.linspace(0.9 * xi_init, 1.1 * xi_init)
plt.plot(Z, surr(Z) - pade_init.eval(Z), label="error")
plt.axvline(x=xi_init, color='black', ls="--", label="xi")
plt.legend()
plt.show()
plt.close()


xi, acc = dual_check([pade_init], s_test, update_surr, 1, 10, tol)
def approx(x):
        sum = 0
        for f in acc:
            sum += f.eval(x)
        return sum
np.vectorize(approx)


Z = np.linspace(0.9 * xi, 1.1 * xi)
plt.plot(Z, surr(Z) - approx(Z), label="error")
plt.axvline(x=xi, color='black', ls="--", label="xi")
plt.legend()
plt.show()
plt.close()