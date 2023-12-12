#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:45:36 2023

Final Project: Lab 6

This lab explores the thermal diffusion of heat across a vertical column of
atmosphere.

The way in which this is done is by the merger of the function from Lab 3,
"atm," and the function from Lab 4, "heat_solve." Additionally, small portions
of Lab 5 were integrated, such as the formatting of the "dalt" vector used
to assign altitudes in which the temperature is assigned (and thus serves as
as an atmospheric layer midpoint boundary).

Upon creating a successful linkage between the two functions, the forward-dif
solver is iterated over and then compared with a recreation of the real Earth
atmosphere. What exactly is different between the two?

Finally a recreation of Venus and Mars are done for the purpose of comparison
to the atmosphere of Earth.

@author: sully
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt


def atm(epsilon=0.255, nlayers=3, sigma=5.67E-8, debug=False, albedo=0.3):
    '''
    

    Parameters
    ----------
    epsilon : float
        Emissivity of our atmospheric layers. The default is 0.255.
        The emissivity is assumed uniform across all layers of n-atmosphere
        unless stated otherwise. Layer 0, the surface of the planet, is
        assumed to be 1 and thus a blackbody.
    nlayers : positive integer, existing as a float
        Amount of layers to our atmosphere. If 1, then np.array is 2 x 2 to
        account for the surface being the first layer (layer[0]) and the
        second being layer[1] = the singular atmospheric layer.
    fluxes : float
        Fluxes takes the place of x in Ax=B, where fluxes are solved for by
        the inverse of A (A**-1) times B.
    sigma : float
       The Stefan-Boltzmann Constant
    albedo: float
        Reflectivity of a planet. Default = 0.3
        For Earth, albedo = 0.3

    Returns
    -------
    Numpy array of temperature of each layer of an n-layer atmosphere.
    
    Amount of layers in the n-layer defined atmosphere, where n>0.

    '''

    # Create an array of coefficients
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    
    # Set all values for 'A' matrix
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i!=j:
                A[i,j]=epsilon*(1-epsilon)**(abs(j-i)-1)
            
            # The "and i!=j" was included so that the expression doesn't come
            # to be 0 to a -1 power when initially looping over
            # the matrix
            
            if i==0 and i!=j:
                A[i,j]=1*(1-epsilon)**(abs(j-i)-1)                
            if i==j:
                A[i,j]=-2        
            if i==j==0:
                A[i,j]=-1
            if debug:
                 print(f'A[i={i}, j={j}] = {A[i,j]}')
                
             
    # Set all values for 'b' matrix
    for i in range(nlayers+1):
        # For nuclear fallout question, "if i!=nlayers:" was used, since the
        # solar influx value would be set equal to what absorbs the visible
        # radiation, thus here being the nth layer.
        
        # For all questions 1-2, the code "if i!=0:" was used
        if i!=0:
            b[i]=0
        else:
            # Interchangable Solar constant
            b[i]=-(1350*(1-albedo)*0.25)
        if debug:
             print(f'b[i={i}] = {b[i]}')
            
            
    # Invert 'A' Matrix
    Ainv = np.linalg.inv(A)
    
    # Get Solution for x in 'Ax=B,' where fluxes == x
    fluxes = np.matmul(Ainv, b) # Note use of matrix multiplication
    
    # Solve for temperature of each layer, with temp[0] having an epsilon = 1
    # for representing the surface of a planet
    temp = ((fluxes)/((sigma)*(epsilon)))**(1/4)
    temp[0] = ((fluxes[0])/((sigma)*(1)))**(1/4)
    
    layercount = nlayers+1
    
    if debug:
        print(f'A[i={i}, j={j}] = {A[i,j]}')
    
    return temp, layercount


def heat_solve(xmax=100.0, dx=1, tmax=86400*365*50, dt=86400, c2=1.9e-11, 
               debug=False, planet = 2):
    '''
    

    Parameters
    ----------
    xmax : float, (units --> kilometers)
         Defines the total length of the one-dimensional space for
        which the forward difference solver is iterated over.
    dx : float (units --> kilometers)
        The spatial step, must be divisible into xmax.
    tmax : float (units --> seconds)
        Defines the total time that the forward difference solver is iterated.
    dt : float (units --> seconds)
        The time step, must be divisible into tmax.
    c2 : float (units --> kilometers squared per second)
        Thermal diffusivity constant, representative of "c squared."
    debug : boolean, defaults to False
        If True, print out debug information.
    
        
    Returns
    -------
    x: numpy vector
        Array of position locations (y-grid)
    t: numpy vector
        Array of time points (x-grid)
    tempL4: 2-dimensional numpy array
        Temperature as a function of time and space

    '''
    
    ###################################################################
    # Question 1
    
    # Call atmosphere function
    tempL3, layercount = atm()
    
    # Convert from the temperature units used in Lab 3 (Kelvin --> Celsius)
    tempL3 += -273
    
    # Create a vector containing values of altitude
    # Each altitude represents an atmospheric layer, of which has a temperature
    # input from the "atm" function
    dalt = xmax / (layercount-1)
    alts = np.linspace(dalt/2, xmax-dalt/2, layercount-1)
    
    ###################################################################
    
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
    tempL4 = np.zeros([M, N])
   
    ################################################################### 
    # Question 2 & 3
    
    # Set initial temperature across the column
    if planet == 3: #Earth
        # Troposphere
        tempL4[:13, 0] = (70/-12)*(x[:13]-2.57)
        # Stratosphere
        tempL4[13:48, 0] = (55/35)*(x[13:48]-47)
        # Stratopause
        tempL4[47:53, 0] = 0
        # Mesosphere
        tempL4[53:86, 0] = (-90/32)*(x[53:86]-53)
        # Mesopause
        tempL4[85:91, 0] = -90
        # Thermosphere up until 100km altitude
        tempL4[91:101, 0] = (15/10)*(x[91:101]-150)
        
    if planet == 2: #Venus
        tempL4[:16, 0] = (-114/15)*(x[:16]-60.8)
        
        tempL4[16:61, 0] = (-358/45)*(x[16:61]-58.75)
        
        tempL4[60:91, 0] = (-94/30)*(x[60:91]-56.8)
        
        tempL4[91:101, 0] = (-8/10)*(x[91:101]+40)
   
   ################################################################### 
    # Question 1, 2, & 3
    
    # Set upper and lower boundary conditions (Dirichlet)
    if planet == 3:
            # The Karman Line @ 100km
        tempL4[-1,:] = -75
            # The temperature of the Earth
                #Question1
        # tempL4[0,:] = tempL3[0]    
                #Question 2
        tempL4[0,:] = 15
        
        #Question 3
    if planet == 2:
        tempL4[-1,:] = -112
        
        tempL4[0, :] = 462
    
    ###################################################################
    # Question 1
    
    # Set "mid-point" boundary conditions
    # These represent the temperature values of the atmospheric layers
    # according to atm function
        # tempL4[int(alts[0]), :] = tempL3[1]
        # tempL4[int(alts[1]), :] = tempL3[2]
        # tempL4[int(alts[2]), :] = tempL3[3]
    
    ###################################################################
    
    # Debug:
    if debug == True:
        print(f"Size of grid is {M} in space, {N} in time")
        print(f"Diffusion coeff = {c2}")
        print(f"Space grid goes from {x[0]} to {x[-1]}")
        print(f"Time grid goes from {t[0]} to {t[-1]}")      
    
    # Solve via the forward-difference solver
    for j in range(0, N-1):
        tempL4[1:-1, j+1] = (1-2*r)*tempL4[1:-1, j] + \
            r*(tempL4[2:,j] + tempL4[:-2,j])
    
    return x, t, tempL4


def plot_func():
    '''
    A function withholding code which makes two separate plots
        - Plot One: Temperature of an "n" layer atmosphere, with layer zero
                    being equivalent to the ground. This ground value is
                    also solved for.
        - Plot Two: Temperature diffusion of a one dimensional space as time
                    passes
                    
    Inputs
    -------
    atm: function
        Provides the output of the linear algebra system of equations solver
    
    heat_solve: function
        Provides the output of the forward difference solver

    Returns
    -------
    Genuinely none, I swear o_o
    
    I mean I suppose this function returns plots. Its not technically a
    return though, right?

    '''
    
    
    # Create plots, add labels, save file
    tempL3, layercount = atm()
    fig, ax = plt.subplots(1,1)
    ax.plot(tempL3, np.arange(layercount), label='Temperature of N-Layer')
    ax.set_title("Temperature of an N-Layer Atmosphere")
    ax.set_xlabel("Temperature (Kelvin)")
    ax.set_ylabel("Layer Number")

    file_counter = 0
    file_name = f'lab6_layerFig_{file_counter:03}.png'
    plt.savefig(file_name)
    
    ###########################################################
    
    # Extract solution from heat_solve function
    x, t, tempL4 = heat_solve()
    
    ###########################################################
    # Plot One #
    
    # Create a figure/axes object for the result of the heat_solve func
    fig, axes = plt.subplots(1, 1)
    
    # Create a color map and add a color bar
            #(x), representative of space, is plotted on the vertical axis
    map = axes.pcolor(((t/86400)/365), x, tempL4, cmap='seismic', vmin=-500, vmax=500)
    plt.colorbar(map, ax=axes, label='Temperature (º$C$)')
    
    # Add plot labels, save plot as file, and display 
    axes.set_title("Heat Diffusion through Layers of the Atmosphere")
    axes.set_xlabel("Time (Years)")
    axes.set_ylabel("Altitude (km)")
    file_counter = 0
    file_name = f'lab6_diffusionFig_{file_counter:03}.png'
    plt.savefig(file_name)
    plt.show()
    
plot_func()