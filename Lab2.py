#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 26 09:31:31 2023

@author: Sullivan Rudolph

This code, if replicated and rerun, will reproduce the results shown and
analyzed in the "Clasp 410 Lab 2 Report"

Lab 2 within the UMich class "Clasp 410 - Earth System Modeling" was performed
to test the varying change two populations had one another by a specific
relationship the two shared. 
    The first - the competition model - explores what would happen if two
    populations existed within the same niche and competed for resources.
    
    The second - the predator-prey model - explores what would happen if
    two populations existed, with one being the only food source for the other.
    
The two populations are assessed by two solvers - the Euler method (first order
accuracy solver) and the RK8 method (eighth order accuracy solver). The
outputted solutions are then plugged into a graph. The x axis represents
time in years (up to 100) and the y axis represents a percent of population
carrying capacity (in decimal form).
    For example, if the Cats (Euler) function gives an output of f(80) = 1.2,
    this means that the population went over carrying capacity during this
    period.
"""

import numpy as np
import matplotlib.pyplot as plt

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    Our competition equation, where N represents two species by a two
    element list. The two species are plugged into each other's function. By
    this mechanism the two have an effect on the population of the other.
    See line 57.
    
    Parameters
    ----------
    t : float
        time, in years
    N : two-element list
        N1 represents population of species 1 (dawgs)
        N2 represents population of species 2 (cats)
    a, b, c, d: float, defualts=1 ,2, 1, 3
        Value of the Lotka-Volterra (competition function) coefficients

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of 'N1' and 'N2' (Competition Function)
    '''
    
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    
    return dN1dt, dN2dt


def dNdt_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    Our prey equation, where N represents two species by a two
    element list. The two species are plugged into each other's function. By
    this mechanism the two have an effect on the population of the other.
    See line 87.

    Parameters
    ----------
    t : float
        time, in years
    N : two-element list
        N1 represents population of species 1 (dawgs)
        N2 represents population of species 2 (cats)
    a, b, c, d: float, defualts = 1, 2, 1, 3
        Value of the Lotka-Volterra (prey function) coefficients

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of 'N1' and 'N2' (Prey Function)
    '''
    
    # Similarly, these functions operate as a two-element list
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = (-c)*N[1] + d*N[1]*N[0]
    
    return dN1dt, dN2dt


def euler_solve (func, N1_init=0.3, N2_init=0.5, dT=0.042, 
                 t_final=100.0, a=1, b=1, c=2, d=2):
    # For competition, use dT = 1
    # For pred/prey, use dT = 0.05
    # This is to prevent too large of an accumulated error (w/ regards to Euler himself)
    
    '''
    Our euler (first order solver). This takes the input of either our
    competition or predator/prey equation. By an estimation of the derivative
    and the value of the function a t=0 where N[0] is given as N#_init,
    Euler can estimate what the function is. However, since Euler is a
    first order solver, this function accumulates inaccuracies much more
    rapdily at t increases relative to the RK8 (eighth order solver).

    Parameters
    ----------
    func : function
        A python function that takes 'time,' ['N1', 'N2'] as inputs and
        returns the time derivative of N1 and N2
    N1_init : float
        The initial population of species N1. The default is 0.5. (50%)
    N2_init : float
        The initial population of species N2. The default is 0.5. (50%)
    dT : float
        Our change in time, shortened from delta T. The default is 0.1.
        Value used for competition model: 1
        Value used for predator-prey model: 0.05
    t_final : float
        Where our models stop. 100 corresponding to 100 years. Default = 100.
    a, b, c, d: float, defaults = 1, 2, 1, 3
        Constant values used within Lotka-Volterra equations. Defaults to
        1, 2, 1, 3 respectively but also have overriding defaults within
        this function

    Returns
    -------
    euler_N1, euler_N2: float
        Euler estimations of the N1 and N2 functions

    '''
    
    # Create time array
    time = np.arange(0, t_final, dT)
    
    # Create solution array
    euler_N1 = np.zeros(time.size)
    euler_N1[0] = N1_init # Set initial population of species N1
    
    euler_N2 = np.zeros(time.size)
    euler_N2[0] = N2_init # Set initial population of species N2
    
    
    for i in range (time.size - 1):
        
        # Assign competition function outputs to a variable. Bypassing
        # the fact that dNdt_comp & dNdt_prey return data in form of a tuple.
        N = func(0, [euler_N1[i], euler_N2[i]], 
                 a=a, b=b, c=c, d=d) 
        dN1dt, dN2dt = N
        euler_N1[i+1] = euler_N1[i] + dT * dN1dt
        euler_N2[i+1] = euler_N2[i] + dT * dN2dt
        
    # Returning the euler estimation of function N1 & N2
    
    return time, euler_N1, euler_N2


def solve_rk8(func, N1_init=0.3, N2_init=0.5, dT=10, t_final=100.0,
              a=1, b=1, c=2, d=2):
    '''
    RK8 eighth order solver. One misconception is that this function doesn't
    accumulate error. It still does, but on a much smaller magnitude. This
    also takes competition or predator/prey as an input.

    Parameters
    ----------
    func : function
        A placeholder to accept the input of a function
        These being either dNdt_comp or dNdt_prey
    N1_init, N2_init : floats
        Initial populations of both species. The default is 0.5.
    dT : float
        time step. The default is 10.
    t_final : TYPE, optional
        The maximimum value of time at which the populations are assessed.
        The default is 100.0. This isn't changed for the lab in any case,
        but feel free to mess with it.
    a, b, c, d : floats, constants
        Lotka-Volterra coefficients. The default is 1, 2, 1, 3
    

    Returns
    -------
    time: Numpy array
        Time elapsed in years
    N1, N2: Numpy arrays
        Normalized population density solutions

    '''
    
    from scipy.integrate import solve_ivp
    
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=[a, b, c, d,], method='DOP853', max_step=dT)
    
    #Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    
    #Return values to caller, I guess
    return time, N1, N2
    

# Creating our plots
fig, ax = plt.subplots(1,1)

# Assign variables for our RK8 solver, assign plot labels
    # Adjust Rk8 & Euler to solve prey or comp by changing func in paranthesis
t, dawgs, cats = solve_rk8(dNdt_comp)
ax.plot(t, dawgs, label='Dawgs (RK8)')
ax.plot(t, cats, label='Cats (RK8)')
    # Phase Diagram -- Placeheld by a # for when not using
# ax.plot(dawgs, cats, label='Dawgs to Cats(RK8)')

# Assign variables for our Euler solver, assign plot labels
t_e, dogs_e, cats_e = euler_solve(dNdt_comp)
ax.plot(t_e, dogs_e, '--', label='Dawgs (Euler)')
ax.plot(t_e, cats_e, '--', label='Cats (Euler)')

# Plot Labeling (X = Time, Y = Population (as decimal))
ax.legend(loc='best')
ax.set_title("Two Species Competing Over Time")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Population (1.0 = 100% of cap)")

# Change save location
# I apologize, I do not know how to have code automatically make a new file
# You must manually adjust the file name each time
fig.savefig(f"/Users/sully/clasp410/Lab2Figures/population-model_{0:02d}.png")

    
    
    