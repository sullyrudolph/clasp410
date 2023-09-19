#!/usr/bin/env python3

"""
Created on Thu Sep  7 10:15:28 2023

@author: sully

This file contains tools and scripts for repeating report generated in Lab 1
for ClaSP 410. To reproduce the plots in the report do this... :)
"""

# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
# Import a color map to color code our object
from matplotlib.colors import ListedColormap


#Variables
nx, ny = 30, 30 # Number of cells in an x and y direction
prob_spread = 0.25 # Chance to spread to adjacent cells
prob_bare = 0.0 # Chance of cell to start as a bare patch
prob_start = 0.2 # Chance of a cell to start on fire


#Initial Conditions

# 1 = bare/burnt, 2 = forest, 3 = burning

outpath = '/Users/sully/clasp410/Lab1Figures'

## Create a grid as a part of the initial conditions. Set all values to "2"
forest = np.zeros([ny, nx], dtype=int) + 2

## Create Colormap
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

## Create figure and set of axes:
fig, ax = plt.subplots(1,1)

## Set center cell to "burning"
forest[16,16] = 3
# forest[7,29] = 3   # If you want to get frisky with the initial conditions
# forest[24,80] = 3

# Pad with -1 Values
forest = np.pad(forest, pad_width=1,mode='constant',constant_values=-1)

idxmask = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])

frames = []


#Functions

def spread_fire(array,center):
    """
    Function to spread a fire to a given quadrant of the cell
    
    Indexes into a given center in a 2D numpy array and assigns the
    surrounding 2 x 2 grid to the values in a different arrray

    Parameters
    ----------
    array : np.ndarray
        2D numpy array representing the state of the forest at given time
    center : tuple
        tuple containing index of central burning cell

    Returns
    -------
    array : np.ndarray
        2D numpy array representing the state of the forest after spreading
        fire at the given burning cell

    """
    # Get new values for burning cells
    repl_array = gen_burn_mask(center[0])
    # Define offsets from central cell for replacement
    offsets = np.array([[-1,-1],[-1,0],[-1,1], [0,-1],[0,0],[0,1], [1,-1],[1,0],[1,1]])
    # Reshapre offset array to allow easy indexing of the 2x2 grid around center
    fill = (offsets + center[:,None]).reshape(-1,2)
    array[fill[:,0],fill[:,1]] = repl_array.flatten()
    return array

def gen_burn_mask(center,mask=idxmask,array=forest,p_spread=prob_spread):
    """
    Generates a mask of new cells representing a "burn pattern"
    at a given location.

    Parameters
    ----------
    center : tuple
        tuple containing coordinates of central index of 3x3 grid around
        the burning cell
    mask : np.ndarray
        2D numpy array containing the spread pattern
    array : np.ndarray
        larger 2D numpy array, here a representation of a forest
    p_spread : float
        probability that the fire will spread to a given cell

    Returns
    -------
    new_vals : np.ndarray
        2D array mask containing new values for fire spread

    """
    
    # Make copy of idx array
    new_mask = np.copy(mask)
    # Get grid surrounding 2 x 2 grid cell
    new_vals = array[center[0]-1:center[0]+2,center[1]-1:center[1]+2]
    # Create mask dictating fire spread by comparing to random 3x3 array
    new_mask[(new_mask == 1) & (np.random.random((3,3)) < p_spread)] = 9 # arbitrary index
    new_vals[(new_mask == 9) & (new_vals == 2)] = 3 # set all values where the mask is burning to 3
    array[center[0],center[1]] = 0 # set center to 0 (burnt on this iteration)
    return new_vals

def plot_frames(frame):
    """
    Creates a pcolor matrix plot representing the state of the 
    forest fire spread in a given forest.

    Parameters
    ----------
    frame : np.ndarray
        2D numpy array representing the state of the forest at one timestep
    
    Returns
    -------
    fig : plt.Figure
        pcolor figure showing the state of the fire at a given time
    """
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    fig, ax = plt.subplots(1,1)
    ax.pcolor(frame, cmap=forest_cmap, vmin=1, vmax=3)
    return fig

def main():
    """
    Main function for execution of forest fire simulation
    """
    # Time code
    st = time.time()
    out = f'{outpath}/gridsize_{nx}_{ny}'
    if not os.path.exists(out):
        os.mkdir(out)
    # Loop over the forest until no more burning cells
    while 3 in forest:
        # Locate index of centers where forest is burning (forest = 3)
        burning_cells = np.asarray(np.where(forest==3)).T.tolist()
        # Loop over burning centers
        for center in burning_cells:
            # spread the fire at each center
            spread_fire(forest,np.array([center]))
        # Append this to a list of frames
        frames.append(plot_frames(forest[1:-1,1:-1]))
    
    # Save all figures
    for idx,frame in enumerate(frames):
        frame.savefig(f'{out}/forest_iter{idx}.png')
        plt.close()
    
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == "__main__":
    main()
