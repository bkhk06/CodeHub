from sympy import *
#from sympy import Symbol, cos

x = Symbol('x')

print((1/cos(x)).series(x, 0, 10))

y = Symbol('y', real=True)
print(exp(I*y).expand(complex=True))

print(log(exp(y)).simplify())

