#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:56:04 2023

@author: sully

This code represents the work used to perform Lab 5. Lab 5 covered the
concept of snowball Earth. This concept was approached in steps.
    1) Validation of numerical solver in steps, which would be later used
        to iterate the temperature models over time.
            - Initial warm Earth condition
            - Thermal diffusion only
            - Addition of spherical correction
            - Addition of insolation and blackbody radiation
    Explore the influence of...
        2) the thermal diffusion and emissivity cosntants
        3) dynamic albedo on temperature curves
        4) insolation

Afterward, a conclusion is made on the possible validity of the snowball
Earth hypothesis.

"""

import numpy as np
import matplotlib.pyplot as plt


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])

    # Used for Question 2 and 4
    # T_warm[:] = -60

    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowEarth(npoints=18, dt=1, tstop=10000, lamb=100, radearth=6357000,
              BaseDiffScenario=False, SphCorrScenario=False, S0=1370, 
              albedo=0.3, emiss=0.725, sigma=5.67e-8, rho=1020, C=4.2e6, dz=50,
              gamma=0.4):
    '''
    

    Parameters
    ----------
    npoints : integer, default is 18
        The amount of points where temperature is recorded 
        across the latitudinal grid of Earth.   
    dt : integer, default is 1
        The time step
        Units: years
    tstop : integer
        The time for which the numerical solver stop. default is 10,000
        Units: years
    lamb : float, default is 100
        lambda constant, representing thermal diffusivity
    radearth: constant
        the radius of the Earth
        Units: meters
    BaseDiffScenario: boolean, default is False
        if True, the numerical solver is executed including only the basic
        diffusion portion of the solver.
    SphCorrScenario: boolean, default is False
        if True, the numerical solver is executed including basic diffusion
        and the spherical correction factor portions of the solver
    S0: solar constant, default is 1370
        Units: watts per square meter
    albedo: float, default is 0.3
        Reflectivity of the surface of Earth
    emiss: float, default 0.725 [after Question 2]
        The emissivity of Earth
    sigma: constant
        Stefan-Boltzmann constant
    rho: constant
        Density of seawater
        Units: kilograms per cubic meter
    C: constant
        Heat capacity of seawater
    dz: constant, default 50 meters
        Mixed-layer depth of ocean
    gamma: float, default 0.4
        Solar-irradiance multiplication factor
        
        
    Returns
    -------
    temp: numpy array
        Returned temperatures in an array with as many points equivalent
        to the inputted amount as "npoints." The returned temperatures will
        adjust depending on the inputted initial temperatures.
    lats: numpy array
        Equivalent in size relative to the inputted value of "npoints" and
        likewise the "temp" array, lats returns an array of latitudes for
        which the temp was recorded, iterated over with the numerical solver,
        and returned for.

    '''
    
    
    # Create a time step   
    nstep = int(tstop / dt)
    
    # Create a space grid (with respect to the latitiudinal coords of Earth)
    dlat = 180 / npoints
    lats = np.linspace(dlat/2, 180-dlat/2, npoints) #degrees
    
    # Set initial condition:
    temp = temp_warm(lats) #ºC

    # Set value of dy
    dy = radearth * np.pi * (dlat / 180) #meters
    
    # Get dt in seconds
    dt_sec = dt * 365*24*3600 #sec
    
    # Set up identity matrix with dimension size equivalent to temperature vector
    I = np.identity(npoints)
    
    ####################################################################
    
    # Set up "A" matrix -- equivalent in size to temperature vector
    A = np.zeros([npoints, npoints])
    
    # Set values of tridiagonal "A" matrix
    for i in range(npoints):
        A[i,i] = -2
    for i in range(npoints-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
        
    # Set values of "A" matrix such that Neumann Boundary conditions are set
    A[0,1] = 2
    A[npoints-1, npoints-2] = 2
        
    # Divide by dy squared to complete the A matrix
    Amat = A / (dy**2) # meters ^ -2
    
    #################################################################### 
       
    # Set up "B" matrix -- equivalent in size to temperature vector
    B = np.zeros([npoints, npoints])
   
    # Set values of tridiagonal "B" matrix
    for i in range(npoints):
        B[i,i] = 0
    for i in range(npoints-1):
        B[i,i+1] = 1
        B[i+1,i] = -1
        
    B[0,1] = 0
    B[npoints-1, npoints-2] = 0
    
    ####################################################################  
    
    # Setting up spherical correction term, which is as follows
    
    Axz = np.pi * ((radearth + 50)**2 - (radearth)**2) * np.sin(np.pi/180. * lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)
    
    # Base diffusion term used to call equation with or without spherical
    # correction term applied to the final solution
    if BaseDiffScenario == True:
        dAxz = 0
    
    ####################################################################
    
    # Initiate the insolation term
    
    insol = gamma * insolation(S0, lats)
    
    ####################################################################
    
    # Create "L" matrix, which is defined below
    L = I - lamb * dt_sec * Amat
    
    # Invert L matrix to be used in solver
    Linv = np.linalg.inv(L)
    
    ####################################################################
    
    # Iterate over time for our temperature matrix
    for i in range(nstep):
        # Perform spherical correction factor
        sphCorr = lamb * dt_sec * np.matmul(B, temp) * dAxz
        
        # Add spherical correction
        temp += sphCorr
        
        # Add insolation
        
            # Add variable albedo values - Used for Question 3
                # albedo = np.zeros([npoints])
                # albedo_ice = 0.6
                # albedo_gnd = 0.3    
            
                # loc_ice = temp <= -10
                # albedo[loc_ice] =  albedo_ice 
                # albedo[~loc_ice] = albedo_gnd
        
        
        radiative = (1-albedo)*insol - emiss*sigma*((temp+273)**4)
        solar = dt_sec * radiative / (rho * C * dz)
        
        if BaseDiffScenario == True or SphCorrScenario == True:
            solar = 0
            
        temp += solar
        
        #Solve
        temp = np.matmul(Linv, temp)
        
    
    return lats, temp

    

# Assign variables & create plots for respective scenarios
lats, tempInit = snowEarth(tstop=0)
    # lats, tempBaseDiff = snowEarth(BaseDiffScenario=True)
    # lats, tempSphCorr = snowEarth(SphCorrScenario=True)
lats, tempInsol = snowEarth()
fig, ax = plt.subplots(1,1)

# Plot the lines of our solver
ax.plot(lats, tempInit, label='Initial Warm Earth Solution')
    # ax.plot(lats, tempBaseDiff, label='Basic Diffusion Only')
    # ax.plot(lats, tempSphCorr, label='Diffusion + SphCorr')
ax.plot(lats, tempInsol, label='Diff + SphCorr + Insol')

# Add plot labels, save plot as file, and display (plot 1) 
ax.set_title("Temperature of Earth at Given Latitudes")
ax.set_xlabel("Latitude (º) (90º = Equator)")
ax.set_ylabel("Temperature (ºC)")
ax.legend(loc='best')
file_counter = 0
file_name = f'lab5_figure0_{file_counter:03}.png'
plt.savefig(file_name)


# Question 4 - Gain average global temperature as a function of gamma
avgEarthTemp = np.sum(tempInsol) / tempInsol.size   

gammaValues = ([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 
                0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

avgEarthTempAscend = ([-77.878, -75.783, -71.052, -66.321, -61.766, -57.493,
                       -53.548, -49.807, -46.251, -42.914, -39.673, -36.611,
                       -33.668, -30.834, -28.102, -25.488, -22.934, -20.460,
                       -18.081, -15.768, -13.505])

avgEarthTempDescend = ([-79.941, -74.617, -69.766, -65.097, -60.8, -56.744,
                        -52.964, -49.298, -45.860, -42.570, -39.413, -36.382,
                        -33.465, -30.654, -27.968, -25.344, -22.827, -20.366,
                        -17.996, -15.692])

fig, ax2 = plt.subplots(1,1)
ax2.plot(gammaValues, avgEarthTempAscend, label='Average Global Temperature, Ascending Gamma')
ax2.plot(gammaValues, avgEarthTempDescend, label='Average Global Temperature, Descending Gamma')

# Add plot labels, save plot as file, and display (plot 2)
ax2.set_title("Average Global Temperature")
ax2.set_xlabel("Solar Multiplier Coefficient (Gamma)")
ax2.set_ylabel("Average Global Temperature (ºC)")
ax2.legend(loc='best')
file_counter = 0
file_name = f'lab5_figure1_{file_counter:03}.png'
plt.savefig(file_name)
plt.show()