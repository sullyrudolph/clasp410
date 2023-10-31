#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:22:47 2023

This lab explores heat diffusion over a one-dimensional space as time passes.
The code and problems executed within this code is centralized around the
forward-difference solver, held within the "heat_solve" function. This
solver is first order in time and second order in space. From this, a plot is
created, where space (x) is plotted on the y-axis and time (t) is plotted on
the x-axis

The two main problems dealt with within this lab are:
    1) A wire where the boundary conditions are preset at 0ºC
        - Really good ice cubes
    2) A study of a vertical column underground in Kangerlussuaq, Greenland.
    How the temperature of this ground changes over the course of years, taking
    into account climate variability and geothermal heating.
    
What are the implications of this system and what can the execution of this
solver provide with regards to real-world insight?

@author: sully
"""

import numpy as np
import warnings
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


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    Function 
        Used to describe the upper boundary condition behavior 
        of the Kangerlussuaq underground column over the course of the year.

    '''
    
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5,
                          3.1, -6.0, -12.0, -16.9])
     
    t_amp = (t_kanger - t_kanger.mean()).max()
   
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()


def heat_solve(xmax=100.0, dx=1, tmax=86400*365*50, dt=86400, c2=2.5e-7, init=0, 
               kanger=temp_kanger):
    '''
    

    Parameters
    ----------
    xmax : float
         Defines the total length of the one-dimensional space for
        which the forward difference solver is iterated over.
    dx : float
        The spatial step, must be divisible into xmax.
    tmax : float
        Defines the total time that the forward difference solver is iterated.
    dt : float
        The time step, must be divisible into tmax.
    c2 : float
        Thermal diffusivity constant, representative of "c squared."
    init : 0 or callable function
        Used to define the initial condition of the temperature profile across
        the one-dimensional space.
    Returns
    -------
    x: numpy vector
        Array of position locations (y-grid)
    t: numpy vector
        Array of time points (x-grid)
    temp: 2-dimensional numpy array
        Temperature as a function of time and space
    dt: float
        Pass through of "dt" as a means to be used later outside of this
        function

    '''
    # Passthrough of dt to be used in the winter max plot
    dtPass = dt

    # Stability Criterion -- Assuming the if statement below is true,
    # a warning will be displayed indicating the resulting plot may and
    # likely will have inaccuracies
    
    
    if (dt > dx**2/(2*c2)):
        warnings.warn('Stability criterion is not met.')
    
    # Set constant value of r
    r = c2 * dt/dx**2
    
    
    # Create space and time grids.
    x = np.arange(0, xmax+dx, dx)
    t = np.arange(0, tmax+dt, dt)
    
    # Save number of points
    M, N = x.size, t.size
    
    # Create temperature solution array
    temp = np.zeros([M, N])
    
    # Set Boundary Conditions (Neumann) -- Use for HW06  
        #  temp[0,0] = temp[1,0]
        #  temp[-1,0] = temp[-2,0]
        
    # Set boundary conditions (Dirichlet - Question 1)
    # temp[0,:] = 0
    # temp[-1,:] = 0     
    
    # Set initial condition
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init
        
    # Set boundary conditions (Dirichlet - Question 2&3)
    if callable(kanger):    
        temp[0,:] = kanger(t)
        
    else:
        temp[0,:] = 0
        
    temp[-1,:] = 5        
    
    # Solve via our forward-difference solver
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + \
            r*(temp[2:,j] + temp[:-2,j])
        # Set Boundary Conditions (Neumann) -- Use for HW06  
            # temp[0,:] = temp[1,:]
            # temp[-1,:] = temp[-2,:]
    
    return x, t, temp, dtPass


# Get solution from solver
x, t, temp, dt = heat_solve()

# Create a figure/axes object
fig, axes = plt.subplots(1, 1)
axes.invert_yaxis()

# Create a color map and add a color bar

# For our plot, time (t) is on the horizontal axis and space (x) is plotted
# on the vertical axis. Temperature is plotted via a color gradient.
map = axes.pcolor(t, x, temp, cmap='seismic', vmin=-25, vmax=25)
plt.colorbar(map, ax=axes, label='Temperature ($C$)')


axes.set_title("Heat Diffusion through Permafrost in Kangerlussuaq, Greenland")
axes.set_xlabel("Time (Seconds)")
axes.set_ylabel("Depth (m)")


# Display the plot and simultaneously save as file
file_counter = 0
file_counter += 1
file_name = f'lab4_figure_{file_counter:03}.png'
plt.savefig(file_name)
plt.show()


# Set indexing for final year of results
loc = int(-365/dt)

# Extract min and max values over final year
winter = temp[:,loc:].min(axis=1)
summer = temp[:,loc:].max(axis=1)

# Create a temperature profile plot
fix, ax2 = plt.subplots(1, 1, figsize=(10,8))
ax2.plot(winter,x,label='Winter')
ax2.plot(summer,x,label='Summer', linestyle='--')
ax2.set_title("Temperature Profile")
ax2.set_xlabel("Time")
ax2.set_ylabel("Space")
plt.show()

'''
Q1 - How would I have an f-string set up to where each time the code is run,
a new file is saved with the file name increasing in number
    Ex: lab4_figure000, lab4_figure001, etc.
    
    - Create a function containing the two plot outputs to separate files
'''