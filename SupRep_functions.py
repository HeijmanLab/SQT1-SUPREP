#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Prof. Dr. Jordi Heijman
Date: 16/012/2024
Script: SupRep SQT1 functions
"""
#%% Import packages

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd
import os 
import myokit
#%%

def cellular_prepace(bcl, pp, cond, AB):
    """
    Pre-pace cellular model according to specified scenarios in the ORD model.
    
    Parameters:
    ----------
    bcl : float
    Basic cycle length.
    
    pp : int
    Number of pre-pacing beats.
  
    cond : int
    Cellular condition:
    - 0: WT (wild type)
    - 1: SQT1 (Short QT Syndrome Type 1)
    - 2: SupRep (SupRep treatment).
    
    AB : int
    Apicobasal gradient region (applicable for `cond=1`):
    - 0: Base
    - 1: Mid
    - 2: Apex.

    Returns:
    -------
    prepaced_states : list
        A list of model state variables after pre-pacing.
    """
    # Load the model
    m = myokit.load_model('MMT/ORD_LOEWE_SQT1.mmt')
    
    # Set the cell mode
    m.set_value('cell.mode', 0)
    
    # Initialize a protocol
    p = myokit.Protocol()
    p.schedule(1, 20, 1, bcl, 0)
    
    # Create a simulation object
    s = myokit.Simulation(m, p)
    
    # Set CVODE solver tolerance.
    s.set_tolerance(1e-8, 1e-8)
    
    # Set the cell-type or condition
    if cond == 0:
        s.set_constant('ikr.cond', 0)   
    elif cond == 1:
        s.set_constant('ikr.cond', 1)   
    elif cond == 2:
        s.set_constant('ikr.cond', 2)
    else:
        raise ValueError(f"Invalid condition value: {cond}. Must be 0 (WT), 1 (SQT1), or 2 (SupRep).")
        
    # Set the sims to experimental set-up
    s.set_constant('extra.Ko', 5)
    
    # Set the Apicobasal gradient
    if cond == 1:
        if AB == 0:  # base
            s.set_constant('ikb.ikb_scalar', 1)
        elif AB == 1:  # mid
            s.set_constant('ikb.ikb_scalar', 1)
        elif AB == 2:  # apex
            s.set_constant('ikb.ikb_scalar', 25)
        else:
            raise ValueError(f"Invalid region value: {AB}. Must be 0 (base), 1 (mid), or 2 (apex).")
    
    # Pre-pace the current cell mode
    s.pre(bcl * pp)
        
    # Save the pre-paced states
    prepaced_states = s.state()

    return prepaced_states

def create_matrix(n, mode="random", seed=None, showit=False, all_sqt=None):
    """
    Create a square matrix with various patterns:
    - Random 50% distribution of zeros and ones ("random").
    - Entirely zeros (all_sqt=False).
    - Entirely ones (all_sqt=True).
    - "targeted": Center square (50% area) with 0, rest with 1.
    Parameters:
    ----------
    n : int
        Size of the matrix (n x n).
    mode : str, optional
        Matrix generation mode.
        Default is "random".
    seed : int, optional
        Random seed for reproducibility in "random" mode.
    showit : bool, optional
        If True, visualize the matrix.
    all_sqt : bool or None, optional
        If True, entire matrix is ones. If False, entire matrix is zeros.
    Returns:
    -------
    matrix : np.ndarray
        A 2D numpy array representing the generated matrix.
    """
    # Handle all_sqt argument
    if all_sqt is True:
        matrix = np.ones((n, n))
    elif all_sqt is False:
        matrix = np.zeros((n, n))
    else:
        matrix = np.ones((n, n))  # Default to all ones
        if mode == "random":
            # Random distribution of zeros and ones
            if seed is not None:
                np.random.seed(seed)
            values = np.random.permutation([0] * (n * n // 2) + [1] * (n * n // 2))
            matrix = values.reshape((n, n))
        elif mode == "targeted":
            square_area = (n * n) // 2
            square_side = int(square_area**0.5)
            while square_side * square_side < square_area:
                square_side += 1
            row_start = (n - square_side) // 2
            row_end = row_start + square_side
            col_start = (n - square_side) // 2
            col_end = col_start + square_side
            matrix[row_start:row_end, col_start:col_end] = 0
        else:
            raise ValueError("Invalid mode. Choose 'random', 'targeted'.")
    
    # Count and print the percentages of zeros and ones
    if showit: 
        unique, counts = np.unique(matrix, return_counts=True)
        total_elements = matrix.size
        for value, count in zip(unique, counts):
            print(f"Value {int(value)}: Count = {count}, Percentage = {count / total_elements * 100:.2f}%")
        # Visualize the matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap='viridis', origin='upper')
        plt.title("Matrix Visualization")
        plt.colorbar(label="Values")
        plt.show()
    return matrix


def set_matrix_states(mat, s, sqt1, wt):
    """
    Set cellular states in a Myokit simulation based on the values in a matrix.

    Parameters:
    ----------
    mat : numpy.ndarray
        A 2D matrix representing the cell states. Each entry determines the cellular condition:
        - 0: Wild Type (WT)
        - 1: Short QT Syndrome Type 1 (SQT1)

    s : myokit.Simulation
        The Myokit simulation object used to set the states.

    sqt1 : list
        A list representing the pre-paced cellular states for the SQT1 condition.

    wt : list
        A list representing the pre-paced cellular states for the WT condition.

    """
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == 0:
                s.set_state(wt, x=j, y=i)
            elif mat[i, j] == 1:
                s.set_state(sqt1, x=j, y=i)


def cellular_prepace_AB(bcl, pp, cond):
    """
    Pre-pace cellular model according to specified scenarios in the ORD model.
    
    Parameters:
    ----------
    bcl : float
    Basic cycle length.
    
    pp : int
    Number of pre-pacing beats.
  
    cond : int
    Cellular condition:
    - 0: WT (wild type)
    - 1: SQT1 (Short QT Syndrome Type 1) base
    - 2: SQT1 (Short QT Syndrome Type 1) apex

    Returns:
    -------
    prepaced_states : list
        A list of model state variables after pre-pacing.
    """
    # Load the model
    m = myokit.load_model('MMT/ORD_LOEWE_SQT1_AB.mmt')
    
    # Set the cell mode
    m.set_value('cell.mode', 0)
    
    # Initialize a protocol
    p = myokit.Protocol()
    p.schedule(1, 20, 1, bcl, 0)
    
    # Create a simulation object
    s = myokit.Simulation(m, p)
    
    # Set CVODE solver tolerance.
    s.set_tolerance(1e-8, 1e-8)
    
    # Set the cell-type or condition
    if cond == 0:
        s.set_constant('ikr.cond', 0)   
    elif cond == 1:
        s.set_constant('ikr.cond', 1)   
    elif cond == 2:
        s.set_constant('ikr.cond', 2)
    else:
        raise ValueError(f"Invalid condition value: {cond}. Must be 0 (WT), 1 (SQT1 base), or 2 (SQT1 apex).")
        
    # Set the sims to experimental set-up
    s.set_constant('extra.Ko', 5)
    
    # Pre-pace the current cell mode
    s.pre(bcl * pp)
    
    # Save the pre-paced states
    prepaced_states = s.state()

    return prepaced_states

def reentry_time(c1, s2, block, interval):
    """
    Determines if reentry occurs and calculates its duration relative to the S2 pulse.

    Parameters:
    ----------
    c1 : float
        The voltage threshold for determining stimulation (e.g., -80 mV).
    
    s2 : int
        The timing of the final S2 stimulus in the simulation (in milliseconds).
    
    block : myokit.DataBlock2d
        The data block containing the membrane potential data in a 2D matrix form (time, x, y).
    
    interval : int
        The time step interval in the simulation (in milliseconds).

    Returns:
    ----------
    pd.DataFrame
        A DataFrame containing:
        - `reentry`: Boolean indicating if reentry occurred.
        - `duration`: The duration of reentry in milliseconds relative to S2.
    """
    # Get data after the final S2 stimulus
    final_stim = int(s2 / interval + 1)
    vm_after_s2 = block.get2d('membrane.V')[final_stim:]  # Data after S2

    # Track the upper-left corner (first grid point in 2D matrix)
    upper_left_corner = vm_after_s2[:, 0, 0]

    # Variables to track repolarization and depolarization phases
    repolarized = False
    depolarized = False

    # Look for depolarization after repolarization at upper-left corner
    for t, vm in enumerate(upper_left_corner):
        current_time = t * interval + s2

        # Check for repolarization (below c1)
        if not repolarized and vm < c1:
            repolarized = True
            print(f"First repolarization at time {current_time} ms.")

        # Check for depolarization (above c1), after repolarization
        elif repolarized and not depolarized and vm > c1:
            depolarized = True
            print(f"First depolarization after repolarization at time {current_time} ms.")

            # Look for reentry ending condition
            for t_end, frame in enumerate(vm_after_s2[t+1:], start=t+1):  # Start from the next time step
                current_end_time = t_end * interval + s2
                if np.all(frame < c1):  # If all elements are below threshold (c1)
                    print(f"Reentry ended at {current_end_time} ms.")
                    return pd.DataFrame({"reentry": [True], "duration": [current_end_time - s2]})

            # If no end condition met, fall through to handle persistent reentry

    # If depolarization occurs but the reentry does not end, calculate relative to the last time point
    if depolarized:
        last_time = vm_after_s2.shape[0] * interval + s2  # Total simulation time
        reentry_duration = last_time - s2  # Duration relative to S2
        print(f"Reentry lasted until the last time point: {last_time} ms.")
        print(f"Reentry duration relative to S2: {reentry_duration} ms.")
        return pd.DataFrame({"reentry": [True], "duration": [reentry_duration]})

    # If no depolarization occurs after repolarization
    print("No reentry detected.")
    return pd.DataFrame({"reentry": [False], "duration": [None]})


def process_reentry_data(directory, mode, s1s2_range, c1, interval):
    """
    Processes the reentry data for a given mode and range of S1S2 intervals, calculates the reentry time,
    adds missing entries as NaN, and exports the results to a CSV file.

    Parameters:
    ----------
    directory : str
        The directory containing the files to process.
    
    mode : str
        The mode to filter files (e.g., 'all_sqt', 'random').
    
    s1s2_range : range
        The range of S1S2 intervals to consider (e.g., range(200, 410, 10)).
    
    c1 : float
        The voltage threshold for determining reentry (e.g., -80 mV).
    
    interval : int
        The time step interval used in the simulation (in milliseconds).

    Returns:
    ----------
    None
        The function processes the data, adds missing entries for S1S2 intervals without data, 
        and exports the resulting DataFrame to a CSV file named 'reentry_times_{mode}.csv'.
    """

    # Create a list to store the results
    results = []

    # Loop through the S1S2 intervals
    for i in s1s2_range:
        # Load the file for the specific S1S2 interval
        file = f'2D_sim_600cells_conduct10_{mode}_{i}.zip'
        file_path = f"{directory}/{file}"
        
        try:
            block = myokit.DataBlock2d.load(file_path)

            # Calculate the reentry time
            rt = reentry_time(c1=c1, s2=i, block=block, interval=interval)
            
            # Add the S1S2 interval directly to the DataFrame
            rt['s1s2'] = i
            results.append(rt)
            print(f"Processed: {file}")
        except Exception as e:
            # If the file doesn't exist or there's an issue, append NaN for this interval
            print(f"File missing or error processing {file}: {e}")
            # Create a DataFrame with NaN values for this missing S1S2 interval
            missing_data = pd.DataFrame({"reentry": [np.nan], "duration": [np.nan], "s1s2": [i]})
            results.append(missing_data)
    
    # Concatenate all DataFrames in the results list into a single DataFrame
    combined_df = pd.concat(results, ignore_index=True)

    # Export to a CSV file with the mode name
    combined_df.to_csv(f"reentry_times_{mode}.csv", index=False)
    print(f"Exported: reentry_times_{mode}.csv")

    # Return the combined DataFrame (optional)
    return combined_df

