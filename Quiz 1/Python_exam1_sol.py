#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:12:48 2020

@author: betsaleel henry
"""
import numpy as np
import matplotlib.pyplot as plt

#<============== FIRST EXERCISE ==================>
x_axis_vector_big = [-50, 0, 50]
y_axis_vector_big = [0, 100, 0]
x_axis_vector_small = [-25, 25]
y_axis_vector_small = [50, 50]
plt.plot(x_axis_vector_big, y_axis_vector_big, 'r', x_axis_vector_small, y_axis_vector_small, 'b')
#<============ SECOND EXERCISE ====================>
from sympy.functions.elementary.miscellaneous import cbrt
from sympy.interactive import printing 
printing.init_printing(use_latex=True) # For better representation 
from sympy import *
import sympy as sp


x, y = sp.symbols('x y')
eq = Eq(x**2 + (y - cbrt((x**2)))**2, 4)
plot_implicit(eq)

#<============ THIRD EXERCISE =====================>


t = sp.symbols('t')

y = sp.Function('y')(t)

diffeq = Eq(4*y + 5*t, y.diff(t))
display(diffeq)

dsolve(diffeq, y)

#<=================== ORIGNIAL ====================>

[t,y] = np.meshgrid(np.arange(-5,5,0.5),np.arange(-5,5,0.5))

dy = 4*y + 5*t
dt = np.ones(np.size(dy))

plt.quiver(t,y,dt,dy)

#<=================== BONUS ======================> 
[t,y] = np.meshgrid(np.arange(-5,5,0.5),np.arange(-5,5,0.5))

dy = 4*y + 5*t
dt = np.ones(np.shape(dy))

plt.quiver(t,y,dt/np.sqrt(1+np.square(dy)),dy/np.sqrt(1+np.square(dy)))
# OR
plt.quiver(t,y,dt/np.sqrt(np.square(dt)+np.square(dy)),dy/np.sqrt(np.square(dt)+np.square(dy)))
