# -*- coding: utf-8 -*-
"""
Coffee Problem in Class
"""

import numpy as np
import matplotlib.pyplot as plt

# Create a time array:
time = np.arange(0, 600, 1)


def solve_temp(time, k=1/300., T_env=20, T_init=90):
    '''
    This function takes an array of times and returns an array of temperatures corresponding to each time
    

    Parameters
    ----------
    time : Numpy an array of times
        Array of time inputs for which you want corresponding temps
    k : float
        Heat transfer coefficient, 1/300 s^-1
    T_env: float
        Ambient air temperature, 25ºC
    T_init
        Initial coffee temperature
        One with cream added at start -- 85ºC
        One without cream added at start -- 90ºC

    Returns
    -------
    temp -- Numpy array
        A list of temperatures corresponding to time "t'"

    '''

    temp = T_env + (T_init - T_env) * np.exp(-k * time)

    return temp


def time_to_temp(T_targ, k=1/300, T_env=20, T_init=90):
    '''
    

    Parameters
    ----------
    T_targ : TYPE
        DESCRIPTION.
    k : float
        Heat transfer coefficient. The default is 1/300.
    T_env : float
        Ambient air temperature. The default is 20.
    T_init : TYPE, optional
        Initial coffee temperature.

    Returns
    -------
    None.

    '''
    
    return (-1/k) * np.log((T_targ - T_env)/(T_init - T_env))

    
    # Solve our coffee question
    T_cream = solve_temp(time, T_init=85)
    T_nocrm = solve_temp(time, T_init=90)
    
    # Get time to drinkable temp
    t_cream = time_to_temp(60, T_init=85) # Add cream right away
    t_nocrm = time_to_temp(60, T_init=90) # Add cream once at 60
    t_smart = time_to_temp(65, T_init=90) # Put cream in at 65 deg
    
    # Create figure and axes objects
    fig, ax = plt.subplots(1,1)
    
    # Plot lines and label
    ax.plot(time, T_nocrm, label='No cream til cool')
    ax.plot(time, T_cream, label='Cream right away')
    
    ax.axvline(t_nocrm, c='red', ls='--', label='No cream: T=60')
    ax.axvline(t_cream, c='blue', ls='--', label='Cream: T=60')
    ax.axvline(t_smart, c='green', ls='--', label='No cream: T=65')
    
    ax.legend(loc='best')
    
    return fig

