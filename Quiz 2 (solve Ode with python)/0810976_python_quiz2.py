#!/usr/bin/env python
# coding: utf-8

# Problem #1


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Part (a)


def dy2d2t(x, t):
    y = x[0]
    dy = x[1]
    xdot = [[], []]
    xdot[0] = dy
    xdot[1] = t**2 + 5 * np.sin(t)
    return xdot


time = np.linspace(0,1,10)
z2 = odeint(dy2d2t,[1, 0],time)




#Part (b) plot the results
plt.plot(time, z2[:, 0], label='First Order')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()


#plot the results
plt.plot(z2[:, 0], z2[:, 1], label='Second order')
plt.xlabel('y')
plt.ylabel('dy(t)/dt')
plt.legend()
plt.show()


# Problem #1


from sympy.interactive import printing
printing.init_printing(use_latex=True) # For better representation
from sympy import *
import sympy as sp


# Part (c)


t = sp.symbols('t') # This how you can define multiple variables

y = sp.Function('y')(t)


diffeq = Eq(y.diff(t, 2), t**2 + 5*sin(t))
display(diffeq)


sol = dsolve(diffeq)

c1, c2 = sp.symbols('C1, C2') #Constant variables

deriv_1 = c1 + c2*t + (t**4/12) - 5 * sin(t)
display(deriv_1)


first_order_deriv = diff(deriv_1, t, 1)
first_order_deriv

second_order_deriv = diff(deriv_1, t, 2)


# Problem 2


#y' = 2*cos(t) + (-t/10)*e**(-t/10)

# Part (a)

x,v = np.meshgrid(np.linspace(0,10,20),np.linspace(-5,5,20))
dx = 1
plt.quiver(x,v,dx,(2 * np.cos(x) + (-x/10) * np.exp(-x/10)) * dx)
plt.show()
plt.show()


# Part (b)


def dydt(y, t):
    dydt = 2*np.cos(t) + (-t/10) * np.exp(-t/10)
    return dydt



t_time = np.linspace(0,10,20)
y_prime = odeint(dydt,1,t_time)


#plot the results
plt.quiver(x,v,dx,(2 * np.cos(x) + (-x/10) * np.exp(-x/10)) * dx)
plt.plot(t_time, y_prime)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
