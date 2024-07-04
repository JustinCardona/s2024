
def bisection_method(func, a, b, tol=1e-7):
    c = (a + b) / 2
    while abs(func(c)) > tol:
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
        print(c)
    return c

def f(x):
    return x

print(bisection_method(f, -1, 2))