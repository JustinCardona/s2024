from aaa import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def dual_check(func, xi_0, tol=1e-12):
    xi_prev = xi_0
    xi_cur = 1.1 * xi_prev

    domain = np.exp(np.linspace(-xi_0, xi_0 + 0.15j * np.pi, 300))
    a = pade(func, domain)
    err = func(xi_cur)
    while err > tol or np.isnan(err):
        xi_prev = xi_cur
        a.fit(xi_prev)
        _, _, z = a.prz()
        xi_cur = np.max(z)
        err = func(xi_cur)
        print(err)
        

F = lambda z: np.tan(np.pi * z / 2)
dual_check(F, 0.01)
