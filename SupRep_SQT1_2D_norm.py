#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Prof. Dr. Jordi Heijman
Date: 16/12/2024
Script: SupRep SQT1 project
"""
#%% Set the directories and load packages
import matplotlib.pyplot as plt
import seaborn as sns
import myokit
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import time
import multiprocessing
import re

# Set your working directory.
work_dir = os.getcwd()
if 'SupRep SQT1' in work_dir:
    os.chdir(work_dir)
else:
    work_dir = os.path.join(work_dir, 'SupRep SQT1')
    os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)
    
# Load custom functions.
from SupRep_functions import cellular_prepace_AB, create_matrix, set_matrix_states, reentry_time, process_reentry_data
#%% Perform cellular pre-pacing for several conditions

# Define the basic cycle length 
bcl = 1000

# Perform cellular prepacing for a thousand beats
WT_cellpp = cellular_prepace_AB(bcl = bcl, pp = 1000, cond = 0)
SQT1_cellpp = cellular_prepace_AB(bcl = bcl, pp = 1000, cond = 1)
#%% Perform tissue pre-pacing for several conditions

def tissue_prepace(n, t, dur, conduct, interval, wt, sqt1, mode="random", all_sqt=None):
    """
    Run a 2D tissue pre-pacing simulation.

    Parameters:
    ----------
    n : int
        Size of the matrix (n x n).
    t : float
        Basic cycle length of pacing.
    dur : float
        Duration of the pre-pacing simulation.
    conduct : float
        Conductance for the simulation.
    interval : float
        Log interval for recording data.
    wt : float
        Wild-type state value.
    sqt1 : float
        SQT1 state value.
    mode : str, optional
        Matrix generation mode:
        - "random": Random distribution of zeros and ones.
        - "targeted": Targeted delivery. 
        Default is "random".
    all_sqt : bool or None, optional
        If True, use an all-SQT1 matrix.
        If False, use an all-WT matrix.
        If None, use the specified matrix generation mode.
        Default is None.

    Returns:
    -------
    result : dict
        A dictionary with keys 'state', 'pre', and 'block' containing simulation results.
    """
    # Load the model.
    m = myokit.load_model('MMT/ORD_LOEWE_SQT1.mmt')

    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)

    # Create the OpenCL simulation.
    s = myokit.SimulationOpenCL(m, p, ncells=[n, n])

    # Indicate the paced cells
    s.set_paced_cells(3, n, 0, 0)

    # Set the conductance and a sufficiently small step size
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    
    # Matrix creation
    if all_sqt is not None:
        # If all_sqt is explicitly True or False, fill the matrix with all 1s or 0s
        mat = create_matrix(n=n, mode="random", seed=1, showit=False, all_sqt=all_sqt)
    else:
        # Generate the matrix based on the mode
        mat = create_matrix(n=n, mode=mode, seed=1, showit=False, all_sqt=None)

    # Assign the correct pre-paced states to the matrix.
    set_matrix_states(mat=mat, s=s, sqt1=sqt1, wt=wt)
    
    # Set the field.
    s.set_field('ikr.cond', mat)

    # Run the 2D simulation.
    pre = s.run(dur, log=['engine.time', 'membrane.V'], log_interval=interval)
    state = s.state()
    block = pre.block2d()

    # Determine the suffix based on mode or all_sqt
    if all_sqt is True:
        suffix = '_all_sqt'
    elif all_sqt is False:
        suffix = '_all_wt'
    else:
        suffix = f'_{mode}'

    # Save the results with the appropriate suffix
    np.save(f'2D_prepace_{n}cells_conduct{conduct}{suffix}.npy', state)
    block.save(f'2D_prepace_{n}cells_conduct{conduct}{suffix}.zip')

    return {'state': state, 'pre': pre, 'block': block}


# Time the duration of the function.
start_time_pp = time.time()

# Run the 2D pre-pace function.
# WT_2D_prepace = tissue_prepace(n = 600, t = 1000, dur = 10000, conduct = 10, interval = 5, 
#                                  wt = WT_cellpp, sqt1_base = SQT1_base_cellpp, sqt1_apex = SQT1_apex_cellpp, 
#                                  mode = 'random', all_sqt = False)

# SQT1_2D_prepace = tissue_prepace(n = 600, t = 1000, dur = 10000, conduct = 10, interval = 5, 
#                                  wt = WT_cellpp, sqt1_base = SQT1_base_cellpp, sqt1_apex = SQT1_apex_cellpp, 
#                                  mode = 'random', all_sqt = True)

# rand_2D_prepace = tissue_prepace(n = 600, t = 1000, dur = 10000, conduct = 10, interval = 5, 
#                                  wt = WT_cellpp, sqt1= SQT1_cellpp, mode = 'random', all_sqt = None)

# targeted_2D_prepace = tissue_prepace(n = 600, t = 1000, dur = 10000, conduct = 10, interval = 5, 
#                                  wt = WT_cellpp, sqt1 = SQT1_cellpp, mode = 'scenario3', all_sqt = None)

# Record end time.
end_time_pp = time.time()

# Calculate duration in minutes.
duration_minutes_pp = (end_time_pp - start_time_pp) / 60

# Print duration in minutes.
print("Duration pre-pacing:", duration_minutes_pp, "minutes")

#%% 

def tissue_sims_parallel(inputs):
    
    # Define the arguments
    n = 600
    t = 1000
    dur = 100
    dur_sim = 1000
    conduct = 10
    interval = 5
    wt = WT_cellpp
    sqt1 = SQT1_cellpp
    mode = 'random'
    all_sqt = None
    
    # Load the model.
    m = myokit.load_model('MMT/ORD_LOEWE_SQT1.mmt')

    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)

    # Create the OpenCL simulation.
    s = myokit.SimulationOpenCL(m, p, ncells=[n, n])

    # Indicate the paced cells
    s.set_paced_cells(3, n, 0, 0)

    # Set the conductance and a sufficiently small step size
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    
    # Matrix creation
    if all_sqt is not None:
        # If all_sqt is explicitly True or False, fill the matrix with all 1s or 0s
        mat = create_matrix(n=n, mode="random", seed=1, showit=False, all_sqt=all_sqt)
    else:
        # Generate the matrix based on the mode
        mat = create_matrix(n=n, mode=mode, seed=1, showit=False, all_sqt=None)

    # Assign the correct pre-paced states to the matrix.
    set_matrix_states(mat=mat, s=s, sqt1=sqt1, wt=wt)
    
    # Set the field.
    s.set_field('ikr.cond', mat)
    
    # Determine the suffix based on mode or all_sqt
    if all_sqt is True:
        suffix = '_all_sqt'
    elif all_sqt is False:
        suffix = '_all_wt'
    else:
        suffix = f'_{mode}'
        
    # Load the pre-pacing state.
    pp = np.load(f'2D_prepace_{n}cells_conduct{conduct}{suffix}.npy')
    s.set_state(pp)
    
    # Run the model for the S1.
    log = s.run(dur, log=['engine.time', 'membrane.V'], log_interval=interval)
    
    # Perform the simulation for 10 seconds
    for i in range(10):
        p2 = myokit.pacing.blocktrain(t, 1, inputs, 1, 1)
        s.set_protocol(p2)
        s.set_paced_cells(n/2, n/2, 0, 0)
    
        log = s.run(dur_sim, log=log, log_interval=interval)
        block = log.block2d()
        
        # If no more electrical activity is detected (i.e., no re-entry), then stop the simulation.
        vm = block.get2d('membrane.V')
        maxval = np.max(vm[-1].flatten())
        if maxval < -50:
            print('no more activity detected')
            break

    block.save(f'2D_sim_{n}cells_conduct{conduct}{suffix}_{inputs}.zip')
    print(f'2D_sim_{n}cells_conduct{conduct}{suffix}_{inputs}.zip')
        
    return dict(log=log, block=block)
WT_250 = tissue_sims_parallel(230)
#%% Parallel computing

if __name__ == '__main__':
    # Record the start time of the script execution.
    start_time = time.time()
    
    # Initialize an empty list to store the final results.
    final_results = []
    
    # Create a list of S1S2 values to iterate over.
    #my_list = list(range(320, 410, 10))
    my_list = [230, 240]
    
    # Determine the number of CPU cores available and create a Pool object with a maximum number of processes.
    # This pool of processes will be used to perform parallel computations on the elements of 'my_list'.
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=min(num_processes, len(my_list)))
    
    # Apply the function 'vulnerability_window_MT' to each element of 'my_list' in parallel.
    # The 'imap' method returns an iterator that yields the results of the function calls in the order of input.
    results = pool.imap(tissue_sims_parallel, my_list)

    # Close the Pool to prevent any more tasks from being submitted to it.
    pool.close()
    
    # Wait for all the worker processes to finish and terminate the Pool.
    pool.join()
    
    # Record the end time of the script execution.
    end_time = time.time()
    
    # Calculate the total time taken for the script execution.
    total_time_taken = end_time - start_time
    
    # If no error occurred during script execution, print the total time taken in seconds.
    print(f"Time taken: {total_time_taken} seconds") 
    
#%% Calculate the reentry time

# Calculate the reentry time for each mode.
wt_rt = process_reentry_data(directory = work_dir, mode = 'all_wt', s1s2_range = range(200, 410, 10), c1 = -80, interval = 5)
sqt1_rt = process_reentry_data(directory = work_dir, mode = 'all_sqt', s1s2_range = range(200, 410, 10), c1 = -80, interval = 5)
random_rt = process_reentry_data(directory = work_dir, mode = 'random', s1s2_range = range(200, 410, 10), c1 = -80, interval = 5)
targeted_rt = process_reentry_data(directory = work_dir, mode = 'targeted', s1s2_range = range(200, 410, 10), c1 = -80, interval = 5)
#%% For visualisation purposes.

# Create matrices and export them.
wt = create_matrix(n = 100, mode = "random", seed =  1, showit = True, all_sqt = False)
sqt1 = create_matrix(n = 100, mode = "random", seed =  1, showit = True, all_sqt = True)
random = create_matrix(n = 100, mode = "random", seed =  1, showit = True, all_sqt = None)
targeted = create_matrix(n = 100, mode = "targeted", seed =  1, showit = True, all_sqt = None)

np.savetxt('wt_matrix.csv', wt, delimiter=",", fmt="%.0f")
np.savetxt('sqt1_matrix.csv', sqt1, delimiter=",", fmt="%.0f")
np.savetxt('random_matrix.csv', random, delimiter=",", fmt="%.0f")
np.savetxt('targeted_matrix.csv', targeted, delimiter=",", fmt="%.0f")

