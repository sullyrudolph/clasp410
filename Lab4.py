#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:22:47 2023

@author: sully
"""

import numpy as np
import matplotlib.pyplot as plt

solution = np.array([[0., 0., 0., 0., 0., 0.,
        0., 0.       , 0.       , 0.       , 0.       ],
       [0.64     , 0.48     , 0.4      , 0.32     , 0.26     , 0.21     ,
        0.17     , 0.1375   , 0.11125  , 0.09     , 0.0728125],
       [0.96     , 0.8      , 0.64     , 0.52     , 0.42     , 0.34     ,
        0.275    , 0.2225   , 0.18     , 0.145625 , 0.1178125],
       [0.96     , 0.8      , 0.64     , 0.52     , 0.42     , 0.34     ,
        0.275    , 0.2225   , 0.18     , 0.145625 , 0.1178125],
       [0.64     , 0.48     , 0.4      , 0.32     , 0.26     , 0.21     ,
        0.17     , 0.1375   , 0.11125  , 0.09     , 0.0728125],
       [0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
        0., 0.       , 0.       , 0.       , 0.       ]])

def sample_init(x):
    '''Simple boundary condition function'''
    return 4*x - 4*x**2

def heat_solve(xmax=1.0, dx=0.2, tmax=0.2, dt=0.02, c2=1.0, init=0):
    '''
    

    Parameters
    ----------
    xmax : TYPE
        DESCRIPTION.
    deltaX : TYPE
        DESCRIPTION.
    tmax : TYPE
        DESCRIPTION.
    deltaT : TYPE
        DESCRIPTION.
    c2 : float, default=1.0
        Thermal diffusivity constant
    xmax : 
    tmax : 
    init : 

    Returns
    -------
    x: numpy vector
        Array of position locations (x grid)
    t: numpy vector
        Array of time points (y-grid)
    temp: 2-dimensional numpy array
        temperature as a function of time and space

    '''
    
    # Set constants
    r = c2 * dt/dx**2
    
    # Create space and time grids
    x = np.arange(0, xmax+dx, dx)
    t = np.arange(0, tmax+dt, dt)
    
    # Save number of points
    M, N = x.size, t.size
    
    # Create temperature solution array
    temp = np.zeros([M, N])
    
    # Set initial and boundary conditions
    temp[0,:] = 0
    temp[-1,:] = 0
    temp[:,0] = 4 * x - 4*(x)**2
    
    # Set initial condition
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init
    
    # Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + \
            r*(temp[2:,j] + temp[:-2,j])
                
    return x, t, temp