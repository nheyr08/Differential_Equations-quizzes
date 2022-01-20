# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:30:36 2020

@author:  Henry
Python quiz 5
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sympy as sp
from sympy.plotting import plot  
from sympy.interactive import printing 
printing.init_printing(use_latex=True)
from numpy import linalg as LA


# a-+++++++++++++++++++++++++++++++++++++++++++++++
A=np.array([[-1/2,1],[-1,-1/2]])
w, v = LA.eig(A)
print("The eigeinvalues are: ")
display(w)
# b-+++++++++++++++++++++++++++++++++++++++++++++++
t=sp.Symbol("t",real=true)
x1=Function("x1")(t)
x2=Function("x2")(t)
 
F=[Eq(x1.diff(t,1),-1/2*x1+x2),Eq(x2.diff(t,1),-x1-1/2*x2)]
display(F)
#solve the equation symbolically
R=dsolve(F,[x1,x2])
display(R)
print("Real value solutions\n")
print(R)

# c-d++++++++++++++++++++++++++++++++++++++++++++ 

Array=np.zeros(shape=(2,500))
t_t=np.ones(500)
t_time=np.arange(500)

#this function return our system.

def Kfc(X, t):
    x1, x2 = X
    return ([-1/2*x1+x2,-x1-1/2*x2])

x1 = np.linspace(-5.0, 5.0, 30)
x2 = np.linspace(-5.0, 5.0, 30)
#meshgrid takes as argument the two arrays we created above with linspace
#Numpy’s meshgrid lets us construct a 2d sample space based upon our arrays
X1, X2 = np.meshgrid(x1, x2)
t = 0

u, v = np.zeros(X1.shape), np.zeros(X2.shape)

NI, NJ = X1.shape
#puts the values to our 2d array u, and v
for i in range(NI):
    for j in range(NJ):
        x = X1[i, j]
        y = X2[i, j]
        yprime = Kfc([x, y], t)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]

Q = plt.quiver(X1, X2, u, v, color='b')

plt.xlabel('Y1')
plt.ylabel('Y2')
#Define the range over which to plot
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title("Fields")
plt.show()
 
#++++++++++++++++++++++++++++++++++++++++++++++++
#this solves the equation out with scipy's odeint and output the solutions
for her in   [5,8,18]:
    tspan = np.linspace(0, 50, 200)
    y0 = [0.0, her]
    ys = odeint(Kfc, y0, tspan)
    
    plt.plot(ys[:,0], ys[:,1]) # determines the path 

plt.xlabel('Y1') #put labels x to our plot
plt.ylabel('Y2')  #put labels y to our plot
#Define the range over which to plot
plt.xlim([-5, 5])
plt.ylim([-5, 5])
Q = plt.quiver(X1, X2, u, v, color='b')
plt.show()
#++++++++++++++++++++++++++++++++++++++++++++++
#phase portrait

for Me in [0,5,7,16,11,55,60,66,78,90,100]:
    tspan = np.linspace(0, 50, 200)
    y0 = [0.0, Me]
    ys = odeint(Kfc, y0, tspan)
    plt.plot(ys[:,0], ys[:,1]) # determines the path 
    plt.xlabel('Y1') #put labels x to our plot
    plt.ylabel('Y2')  #put labels y to our plot
#Define the range over which to plot
plt.xlim([-5, 5])
plt.ylim([-5, 5])
#Q = plt.quiver(X1, X2, u, v, color='b')
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++  
#this part solves my equation numerically
for him in [16,11]:
    x =np.array( [0.0, him]) 
    h = 0.08
    #this attay stores our matrix
    A = np.array([[-1/2,1],[-1, -1/2]])
    n_step = 500#nstep is some of the stufwe have....
    t = np.zeros(n_step+1)#this stores an array of zeros for our calculation later
    traj = np.zeros(shape=(2,n_step+1))#this create another 2d array 
    traj[:,0] = x #we put our initial condirions to this indexes
     #prints details

    for i in range(n_step):
        traj[:,i+1] = traj[:,i]+ np.dot(A,traj[:,i])*h
        t[i+1] = t[i]+h
    #this uses euler technique to qproximate curve   
    plt.plot(traj[0,:],traj[1,:])
    plt.title("2 Numerical solution")
    plt.xlim([-5, 5]) #sets our boundaries
    plt.ylim([-5, 5]) 

plt.show() #shows the final plot
# ++++++++++++++++++++++++++++++++++++++++++++++
for us in   [5,7,11,16]:
    tspan = np.linspace(0, 50, 200)
    y0 = [0.0, us]
    ys = odeint(Kfc, y0, tspan) 
    plt.plot(tspan,ys[:,0])
    plt.plot(tspan,ys[:,1])
    plt.xlim([-6, 6]) #sets our boundaries
    plt.title("4 solutions/1")
    plt.ylim([-5, 5])
plt.show() 
for her in   [5,7,11,16]:
    tspan = np.linspace(0, 50, 200)
    y0 = [0.0, her]
    ys = odeint(Kfc, y0, tspan)
    
    plt.plot(ys[:,0], ys[:,1]) # determines the path 

plt.xlabel('Y1') #put labels x to our plot
plt.ylabel('Y2')  #put labels y to our plot
#Define the range over which to plot
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title("4 solutions/2")
Q = plt.quiver(X1, X2, u, v, color='b')
plt.show() 

''' REPORT_____
1) What is the problem about?
using python(numpy,scipy and sympy) to solve and plot systen of differential equations 
direction fields,phase portrait, numerically, symbollicaly...
2) How did you approach to solve the problem?
I use the techniques taught online to solve the problem, and in documentations to approach the problem
 
3) What difficulties you faced and how did you overcome them?
    debugging issues,Not sure if plotted enough graphs, Tas not availble ...
4) What did you learn from this quiz?
    try to debug with other midterms to prepare can be frustrating.
    in overall I think I learn more things than the other quizzes combined, so its a positive thing, since we have more time this time
    using the tools for solving a system of equation is not so different from one ODE, 
    using python to get eigenvalue/ vector... real value solution finding method, ploting phase portrait and so on .
    I try to add some comments to my code for more details.

'''