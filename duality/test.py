from aaa import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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
        

Z = np.exp(np.linspace(-0.5, 0.5 + 0.15j * np.pi, 1000))
F = lambda z: np.tan(np.pi * z / 2)

a = pade(F, "aaa.pickle")
#a = pade(F, Z)

plt.plot(Z, a.eval(Z) - F(Z))
plt.show()
plt.close()

a.fit(0.5)
a.cleanup(1e-7)

plt.plot(Z, a.eval(Z) - F(Z))
plt.show()
plt.close()
