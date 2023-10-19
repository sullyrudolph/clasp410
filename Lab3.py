#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 5 10:36:18 2023

This code represents all work done relative to Lab 3. If replicated,
the user should be able to reproduce the results analyzed within the
lab report.

Goal: Solve a linear algebra function in the form of Ax = B, where the values
of the A matrix and B matrix are both known, thusly solving for 'x' by the
means of x = (A)**-1 * b

This lab explores the vertical temperature profile of an atmosphere,
represented on the plot starting with the greatest number layer at the top
of the atmosphere, and the lowest number, layer 0, being the surface of the
planet. 

@author: sully
"""

import numpy as np
import matplotlib.pyplot as plt


def atm(epsilon=0.5, nlayers=5, sigma=5.67E-8, debug=False, albedo=0.0):
    '''
    

    Parameters
    ----------
    epsilon : float
        Emissivity of our atmospheric layers. The default is 0.5.
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
        For Earth I used albedo = 0.3
        For Venus I used albedo = 0.77
        For fallout question I used albedo = 0.0
        
        Used bond albedo from NASA, since no value was 
        given to use on the lab worksheet.

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
                
            #Used for the nuclear fallout question (top layer becomes blackbody)
            if i==nlayers and i!=j:
                 A[i,j]=1*(1-1)**(abs(j-i)-1) 
                 
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
        if i!=nlayers:
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


# Create plots, add labels, save file
temp, layercount = atm()
fig, ax = plt.subplots(1,1)
ax.plot(temp, np.arange(layercount), label='Temperature of N-Layer')
ax.set_title("Temperature of an N-Layer Atmosphere")
ax.set_xlabel("Temperature (Kelvin)")
ax.set_ylabel("Layer Number")

# Once again, I am unable to have the file name adjust automatically. I
# apologize. I'll get this figured out for Lab 4.
fig.savefig("/Users/sully/clasp410/Lab3Figures/Nuke_5layer_0.5emiss.png")