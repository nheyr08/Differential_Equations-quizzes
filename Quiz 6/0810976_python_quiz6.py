# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:52:56 2020

@author: Betsaleel Henry
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.pyplot as plt
from sympy import *
import sympy as sp
from sympy.interactive import printing 
printing.init_printing(use_latex=True) 
 
a0=1
a1=1
k=10
n=np.arange(k)
m = 100
rangee = 9.5
x = np.linspace(0, rangee, m)

Y1=np.zeros(k)
Y2=np.zeros(k)
Y=np.zeros(shape=(2,m))
yx=np.zeros(k)

def factorial(l):
    if l == 0:
        return 1
    else:
        return l * factorial(l-1)

def Myfunc(x,n):
   K1=0
   K2=0
   for n in range(k):
        Y1=a0*((((-1)**n)*x**(2*n))/(factorial(2*n)))
        Y2=a1*(((-1)**n)*x**(2*n+1)/(factorial((2*n)+1)))
        K1+=Y1
        K2+=Y2
   return K1,K2
 
for j in range(m): 
       Y[0][j], Y[1][j]= Myfunc(x[j],0)  
     

plt.plot(x,Y[0,:])
plt.xlabel('Figure 1')
plt.show()

plt.plot(x,Y[1,:])
plt.xlabel('Figure 2')
plt.show()

z = np.sin(x)
O = np.cos(x)

plt.plot(x,Y[0,:])
plt.plot(x,O)
plt.xlabel('Figure 3')

plt.legend(['True(cos)','Approx'])
plt.show()

plt.plot(x,Y[1,:])
plt.plot(x,z)
plt.legend(['True(sin)','Approx'])
plt.xlabel('Figure 4')
plt.show()
