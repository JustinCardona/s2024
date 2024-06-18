from aaa import *
from scipy.optimize import fsolve

def dual_check(func, xi_0, tol=1e-5):
    xi_prev = xi_0
    xi_cur = 1.1 * xi_prev
    domain = np.array([0.9 * xi_prev, xi_prev])
    a = pade(func, domain)
    
    err = a.eval(xi_cur)
    print(err)
    while err > tol or np.isnan(err):
        xi_prev = xi_cur
        a.fit(xi_prev)
        xi_cur = fsolve(a.eval, xi_cur)[0]        
        err = a.eval(xi_cur)
        print(xi_prev, err)
        

Z = np.exp(np.linspace(-0.5, 0.5 + 0.15j * np.pi, 100))
F = lambda z: np.tan(np.pi * z / 2)

dual_check(F, 0.01)