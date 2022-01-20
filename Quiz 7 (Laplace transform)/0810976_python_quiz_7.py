# -*-  
"""
Created on Wed Dec 23 15:15:23 2020

@author: wilso
"""

from scipy import signal as sg
import sympy as sp
from sympy import Heaviside
sp.init_printing()
import matplotlib.pyplot as plt
from sympy import DiracDelta as DD
from sympy.plotting import plot 
from sympy.integrals.transforms import laplace_transform
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import *
import sympy as sympy
from sympy.abc import a, t, x, s, X, g, G
init_printing(use_unicode=True)
#x, g, X = symbols('x g X', cls = Function)
t = sp.Symbol('t')
s = sp.Symbol('s')
a = sp.Symbol('a', real=True, positive=True)
F= sp.Function('F')

def L(f):
    return sp.laplace_transform(f, t, s, noconds=True)
def invL(F):
    return sp.inverse_laplace_transform(F, s, t)

#***************************Question1**********************
f = 6*sp.exp(-3*t)-4*sp.cos(5*t)
k=sp.integrate(f*sp.exp(-s*t), (t, 0, sp.oo)).simplify()
print('\n')
Me=L(f)
print('THis is first question\n',Me)
display(Me)
print('\n')

#***************************Question2**********************
#inverse laplace transform
BMW= (6+2*s)/s**4
Lambo=invL(BMW)
print('THis is second question\n')
display(Lambo)
print('\n')

#***************************Question3**********************
#laplace transform
JFK=sp.Piecewise((1,(t>=0)&(t<2)),(2*t-3,(t>=2)&(t<4)),(3,t>=4))
#JFK=sp.Piecewise((1,((0>=t)&(t<2)),((2t − 3),((2>=t)&(t<4))),(3,((4<=t)))))
k=sp.integrate(JFK*sp.exp(-s*t), (t, 0, sp.oo)).simplify()
print('\n')
Me=L(JFK)
print('THis is third question\n',Me)
display(Me)
print('\n')

#***************************Question4**********************
#initial conditions
Y0=0.5
V0=3
Nemo=sp.Eq((s**2*F(s)-s*Y0-V0)+2*(s*F(s)-Y0)+11*F(s),2/s+3*sp.exp(-5*s))
#print('Nemo:',Nemo)
BB= sp.solve(Nemo,F(s)) 
print(BB)
Ket=inverse_laplace_transform(BB[0],s,t)
Ket
plot(Ket, line_color='red',xlim=(0, 10))
#***************************Question5**********************
Bef=sp.laplace_transform(Heaviside(s-a), a, s, noconds=True)
display(Bef)
print(Bef)
