#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:04:02 2020

@author: anvarkunanbaev
"""
#<============= TASK 1 ==================>
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sympy as sp
from sympy.interactive import printing 
printing.init_printing(use_latex=True) # For better representation 

def quiz_3(y0, a, b, r, p):
    t = sp.symbols('t')
    y = sp.Function('y')

    yt = sp.Eq(y(t).diff(t, 1), r * b - ((r/a) * y(t)))
    yt = sp.dsolve(yt, ics={y(0): y0})

    y_inf = sp.limit(yt.rhs, t, sp.oo)

    if y0 >= y_inf:
        tim = -a/r * sp.log(((1 + p) * y_inf - a * b)/ (y0 - a * b))
    else:
        tim = -a/r * sp.log(((1 - p) * y_inf - a * b)/ (y0 - a * b))


    x_max = tim + 3
    y_max = max(y0, y_inf) + 20

    sp.plot(yt.rhs, xlim=(0, x_max), ylim=(0, y_max))

    return yt, y_inf, tim

a, b, r, y0, p = 80, 0.4, 35, 10, 0.03 
y_t, y_infnt, time = quiz_3(y0, a, b, r, p)

print(y_t)
print(y_infnt)
print(time)
#<============= TASK 2 ==================>
from scipy.integrate import odeint

x, y = np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
dx = 1
plt.quiver(x, y, dx, (4 * y + 5 * x) * dx)
plt.show()
plt.show()

def dydt(y, t):
    dydt = 4 * y + 5 * t
    return dydt



t_time = np.arange(0,0.9,0.01)
t_time_2 = np.arange(-0,0.5,0.01)
y_prime = odeint(dydt,-4/16,t_time)
y_prime_2 = odeint(dydt,-5/16,t_time_2)


#plot the results
plt.quiver(x, y, dx, (4 * y + 5 * x) * dx)
plt.plot(t_time, y_prime, label="-4/16")
plt.plot(t_time_2, y_prime_2, label="-5/16")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()


def odeEuler(f,y0,t):
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0,len(t)-1):
        y[n+1] = y[n] + f(y[n],t[n])*(t[n+1] - t[n])
    return y


t = np.arange(0,1,0.01)
y0 = -4/16
f = lambda y,t: (4 * y + 5 * t)
y_prime_3 = odeEuler(f,y0,t)
plt.figure(figsize=(8,8))
plt.quiver(x, y, dx, (4 * y + 5 * x) * dx)
plt.plot(t,y_prime_3,'b.-',t_time,y_prime,'r-')
plt.legend(['Euler','True'])
plt.axis([-1,1,-1,1])
plt.title("Solution of $y'=4y + 5t , y(0)=-4/16$")
plt.show()

t = np.arange(-0,0.5,0.01)
y0 = -5/16
f = lambda y,t: (4 * y + 5 * t)
y_prime_4 = odeEuler(f,y0,t)
plt.figure(figsize=(8,8))
plt.quiver(x, y, dx, (4 * y + 5 * x) * dx)
plt.plot(t,y_prime_4,'b.-',t_time_2,y_prime_2,'r-')
plt.legend(['Euler','True'])
plt.axis([-1,1,-1,1])
plt.title("Solution of $y'=4y + 5t , y(0)=-5/16$")
plt.show()

#<============= TASK 3 ==================>
x = np.arange(-3, 3, 0.1)
y = np.arange(-3, 3, 0.1)


X, Y = np.meshgrid(x, y)

def f(x, y):
    a = x + y * 1j
    return a**2 + 4

Z = f(X, Y)
fig1 = plt.figure(1, figsize=(8,8))
ax1 = plt.axes(projection='3d')
ax1.contour3D(X, Y, Z, 50, cmap='binary')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')


fig2 = plt.figure(2, figsize=(8,8))
ax2 = plt.axes(projection='3d')
ax2.plot_wireframe(X, Y, Z, color='black')
ax2.set_title('wireframe')


fig3 = plt.figure(3, figsize=(8,8))
ax3 = plt.axes(projection='3d')
ax3.plot_surface(X, Y, Z.real, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none') #Use REAL part
ax3.set_title('surface')