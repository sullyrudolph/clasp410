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

# Solution to test for code validity. References specifically to question 1
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
    '''Simple boundary condition function. Used for setting the initial
    temperature profile across the vertical column of space in Q1.
    '''
    return 4*x - 4*x**2


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.

    Parameters
    ----------
    t : float (unit --> days)

    Returns
    -------
    Function 
        Used to describe the upper boundary condition behavior 
        of the Kangerlussuaq underground column over the course of the year.

    '''
    
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5,
                          3.1, -6.0, -12.0, -16.9])
     
    t_amp = (t_kanger - t_kanger.mean()).max()
   
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + 3


def heat_solve(xmax=100.0, dx=1, tmax=86400*365*80, dt=86400, c2=2.5e-7, init=0, 
               kanger=temp_kanger, debug=False):
    '''
    

    Parameters
    ----------
    xmax : float, (units --> meters)
         Defines the total length of the one-dimensional space for
        which the forward difference solver is iterated over.
    dx : float (units --> meters)
        The spatial step, must be divisible into xmax.
    tmax : float (units --> seconds)
        Defines the total time that the forward difference solver is iterated.
    dt : float (units --> seconds)
        The time step, must be divisible into tmax.
    c2 : float (units --> meters squared per second)
        Thermal diffusivity constant, representative of "c squared."
    init : 0 or callable function
        Used to define the initial condition of the temperature profile across
        the one-dimensional space.
    kanger: callable function
        Used to import the temp_kanger function and apply it to the upper
        boundary condition
    debug : boolean, defaults to False
        If True, print out debug information.
        
    Returns
    -------
    x: numpy vector
        Array of position locations (y-grid)
    t: numpy vector
        Array of time points (x-grid)
    temp: 2-dimensional numpy array
        Temperature as a function of time and space
    dt: float
        Pass through of "dt" as a means to use its value later outside of this
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
    
    # Debug time:
    if debug:
        print(f"Size of grid is {M} in space, {N} in time")
        print(f"Diffusion coeff = {c2}")
        print(f"Space grid goes from {x[0]} to {x[-1]}")
        print(f"Time grid goes from {t[0]} to {t[-1]}")
    
    # Set initial condition
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init
        
    # Set boundary conditions (Dirichlet - Question 2&3)
    if callable(kanger):    
            # Since kanger_temp accepts t in units of days, here t is converted
            # from seconds to days
        temp[0,:] = kanger(t/(24*3600.))
        
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


def plot_func():
    '''
    A function withholding code which makes two separate plots
        - Plot One: Temperature diffusion of a one dimensional space as time
                    passes
        - Plot Two: Temperature profile of the vertical column. The coldest
                    each point reaches is represented by "Winter" and the
                    warmest each point reaches is represented by "Summer"
                    
    Inputs
    -------
    heat_solve: function
        Provides the output of the forward difference solver

    Returns
    -------
    None, fr (for real)

    '''
    
    # Get solution from solver
    x, t, temp, dt = heat_solve()
    
    ###########################################################
    # Plot One #
    
    # Create a figure/axes object for the result of the forward-diff solver
    fig, axes = plt.subplots(1, 1)
    axes.invert_yaxis()
    
    # Create a color map and add a color bar
            # One thing to note is (x), representative of space, is plotted
            # on the vertical axis
    map = axes.pcolor(((t/86400)/365), x, temp, cmap='seismic', vmin=-5, vmax=5)
    plt.colorbar(map, ax=axes, label='Temperature (º$C$)')
    
    # Add plot labels, save plot as file, and display 
    axes.set_title("Heat Diffusion through Permafrost in Kangerlussuaq, GL")
    axes.set_xlabel("Time (Years)")
    axes.set_ylabel("Depth (m)")
    file_counter = 0
    file_name = f'lab4_figure_{file_counter:03}.png'
    plt.savefig(file_name)
    plt.show()
    
    ###########################################################
    # Plot Two #
    
    # Set indexing for final year of results
    loc = int(-365/(dt/86400))
    
    # Extract min and max values over final year
    winter = temp[:,loc:].min(axis=1)
    summer = temp[:,loc:].max(axis=1)
    
    # Create a temperature profile plot
    fix, ax2 = plt.subplots(1, 1, figsize=(10,8))
    ax2.plot(winter,x,label='Winter')
    ax2.plot(summer,x,label='Summer', linestyle='--')
    ax2.invert_yaxis()
    ax2.legend(loc='best')
    ax2.set_title("Vertical Profile of the Final Year's Max and Min Temperatures")
    ax2.set_xlabel("Temperature (ºC)")
    ax2.set_ylabel("Depth of Vertical Column (m)")
    plt.axvline(x =0, linestyle = '--', color = 'k')
    file_counter = 4
    file_name = f'lab4_figure_{file_counter:03}.png'
    plt.savefig(file_name)
    plt.show()

plot_func()

''' Notes of Stuff to Add
        1) Dashed vertical line at 0ºC for Plot 2'''