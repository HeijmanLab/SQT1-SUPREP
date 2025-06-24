#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Prof. Dr. Jordi Heijman
Date: 10/12/2024
Script: SQT1 SupRep
"""
# %%
# Load the packages
import matplotlib.pyplot as plt
import seaborn as sns
import myokit
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import time
import multiprocessing

# Set your working directory.
work_dir = os.getcwd()
if 'SQT1_SupRep' in work_dir:
    os.chdir(work_dir)
else:
    work_dir = os.path.join(work_dir, 'Documents', 'GitHub', 'SQT1_SupRep')
    os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

# Load the functions
#%% Load the model and set the experimental conditions.

# Load the model.
m = myokit.load_model('MMT/ORD_LOEWE_SQT1.mmt')

# Set the cell modes to endocardial.
m.set_value('cell.mode', 0)

# Get pacing variable.
pace = m.get('engine.pace')

# Remove binding to pacing mechanism before voltage coupling.
pace.set_binding(None)

# Get membrane potential.
v = m.get('membrane.V')
# Demote v from a state to an ordinary variable; no longer dynamic.
v.demote()
# right-hand side setting; value doesn't' matter because it gets linked to pacing mechanism.
v.set_rhs(0)
# Bind v's value to the pacing mechanism.
v.set_binding('pace')

# Get intracellular potassium.
ki = m.get('potassium.Ki')
# Demote ki from a state to an ordinary variable; no longer dynamic.
ki.demote()
# Set rhs to 135 according to experimental set-up
ki.set_rhs(135)

#%% Voltage-clamp protocol

## Initialize the steps in patch-clamp protocol and add a small margin due to problems 
## w/ zero division.
steps = np.arange(-31, 30, 10) 
p = myokit.Protocol()
for k,step in enumerate(steps):
    # 1500 ms of holding potential
    p.add_step(-80, 1500) 
    # Voltage step to -40 mV for 50 ms
    p.add_step(-40, 50)
    # Voltage step for 1000 ms
    p.add_step(step, 1000) 
    # Voltage step back to -40 mV for 2000 ms
    p.add_step(-40, 2000)
    # Resume holding potential for 1500 ms
    p.add_step(-80, 1500)
t = p.characteristic_time()-1
s = myokit.Simulation(m, p)

def volt_clamp(cond = 0, AB = 0):
    
    # Reset protocol.
    s.reset()
    
    # Re-initiate the protocol.
    s.set_protocol(p)
    
    # Set the maximum stepsize to 2ms to obtain better step curves.
    s.set_max_step_size(1) 
     
    # Set the extracellular potassium concentration to the experimental conditions.

    # Set tolerance to counter suboptimal optimalisation with CVODE.
    s.set_tolerance(1e-8, 1e-8)

    # Set the cell-type or condition (0 = WT, 1 = SQT1, 2 = SupRep)
    if cond == 0:
        s.set_constant('ikr.cond', 0)   
    elif cond == 1:
        s.set_constant('ikr.cond', 1)   
    elif cond == 2:
        s.set_constant('ikr.cond', 2)
    else:
        raise ValueError(f"Invalid condition value: {cond}. Must be 0 (WT), 1 (SQT1), or 2 (SupRep).")
        
    # Set the AB gradient
    if cond == 1:
        if AB == 0:  # base
            s.set_constant('ikb.ikb_scalar', 1)
        elif AB == 1:  # mid
            s.set_constant('ikb.ikb_scalar', 10)
        elif AB == 2:  # apex
            s.set_constant('ikb.ikb_scalar', 25)
        else:
            raise ValueError(f"Invalid region value: {AB}. Must be 0 (base), 1 (mid), or 2 (apex).")
    
    # Run the simulation protocol and log several variables.
    d = s.run(t)

    # Split the log into smaller chunks to overlay; to get the individual steps.
    ds = d.split_periodic(6050, adjust=True) 

    # Initiate the peak current variable.
    Ikr_steady = np.zeros(len(ds)) 
    
    # Trim each new log to contain the steps of interest by enumerating through the individual duration steps.
    for k, d in enumerate(ds):
        # Adjust is the time at the start of every sweep which is set to zero.
        steady = d.trim_left(1551, adjust = True) 
        
        # Duration of the peak/steady current, shorter than max duration to prevent interference between steady peak and upslope of tail.
        steady = steady.trim_right(999) 
        
        # Obtain the max of the steady. 
        Ikr_steady[k] = max(steady['ikr.IKr'])

    return dict(steady = Ikr_steady, d = d)
    
wt_volt = volt_clamp(cond = 0, AB = 0)
sqt1_base_volt = volt_clamp(cond = 1, AB = 0)
suprep_volt = volt_clamp(cond = 2, AB = 0)

plt.figure()
plt.plot(steps, wt_volt['steady'], label = 'WT')
plt.plot(steps, sqt1_base_volt['steady'], label = 'SQT1')
plt.plot(steps, suprep_volt['steady'], label = 'SupRep')
plt.title('Steady IKr currents')
plt.ylabel('IKr [pA/pF]')
plt.xlabel('Membrane potential [mV]')
plt.legend()
plt.tight_layout()

#%% Action potentials 

# Load the model.
m = myokit.load_model('MMT/ORD_LOEWE_SQT1.mmt')

# Set the cell modes to endocardial.
m.set_value('cell.mode', 0)

# Create an action potential protocol.
pace = myokit.Protocol()

# Set the basic cycle length to 1 Hz.
bcl = 1000

# Create an event schedule.
pace.schedule(1, 20, 0.5, bcl)

def action_pot(m, p, bcl, prepace, cond = 0, AB = 0):
    """
    Action potential effects
    
    This script performs a simulation of action potential effects using a given model and action potential protocol. It calculates the action potential duration (APD) and other relevant data based on the specified parameters.
    
    Parameters:
    ----------
    m : myokit.Model
        The Myokit model representing the cellular electrophysiology.
    
    p : myokit.Protocol
        The action potential protocol to be applied during the simulation.
    
    x : list of float
        List of parameter values to be set for the iKr (rapid delayed rectifier potassium current) model components.
    
    bcl : int
        The basic cycle length in milliseconds (ms) used in the action potential protocol.
    
    prepace : int
        The number of pre-pace cycles to stabilize the model before starting the simulation.
    
    
    mt_flag : bool, optional (default = True)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'mt' condition.
    
    carn_flag : bool, optional (default = False)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'carn' condition.
        
    apex : int
        Flag indicating whether the simulation should consider the base (0), mid (1) or apex (2) region. 
        
    Returns:
    -------
    dict
        A dictionary containing the following elements:
        - 'data': The simulation data, including time, membrane potential (V), and iKr current (IKr).
        - 'apd': A myokit.APDMeasurement object representing the action potential duration data.
        - 'duration': The calculated action potential duration (APD90) in milliseconds (ms).
        - 'ikr': The iKr current data obtained from the simulation.
    
    Note:
    -----
    - The simulation will adjust the iKr current based on the provided 'mt_flag' and 'carn_flag' conditions.
    - The CVODE solver tolerance is set to 1e-8 for numerical stability.
    - The action potential is pre-paced for a specified number of cycles ('prepace') to stabilize the model.
    - The action potential duration (APD) is calculated using a threshold of 90% repolarization (APD90).
    """

    # Create a simulation object.
    sim = myokit.Simulation(m, p)
    
    # Set CVODE solver tolerance.
    sim.set_tolerance(1e-8, 1e-8)
    
    # Set the cell-type or condition
    if cond == 0:
        sim.set_constant('ikr.cond', 0)   
    elif cond == 1:
        sim.set_constant('ikr.cond', 1)   
    elif cond == 2:
        sim.set_constant('ikr.cond', 2)
    else:
        raise ValueError(f"Invalid condition value: {cond}. Must be 0 (WT), 1 (SQT1), or 2 (SupRep).")
        
    # Set the sims to experimental set-up
    sim.set_constant('extra.Ko', 5)
    
    # Set the AB gradient
    if cond == 1:
        if AB == 0:  # base
            sim.set_constant('ikb.ikb_scalar', 1)
        elif AB == 1:  # mid
            sim.set_constant('ikb.ikb_scalar', 10)
        elif AB == 2:  # apex
            sim.set_constant('ikb.ikb_scalar', 25)
        else:
            raise ValueError(f"Invalid region value: {AB}. Must be 0 (base), 1 (mid), or 2 (apex).")
    
    # Pre-pace the model.
    sim.pre(prepace * bcl)
    
    # Run the simulation and calculate the APD90.
    vt = 0.9 * sim.state()[m.get('membrane.V').index()]
    data, apd = sim.run(bcl, log=['engine.time', 'membrane.V', 'ikr.IKr'], 
                        apd_variable='membrane.V', apd_threshold=vt)
    
    # Get IKr out of the simulation.
    ikr = data['ikr.IKr']
    
    # Determine the APD duration.
    duration = round(apd['duration'][0], 2)    
    
    return dict(data=data, apd=apd, duration=duration, ikr=ikr)

# Only the SQT1 model has an AB gradient where the apex is 15 ms faster than the base. 
sqt1_ap_base = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 1, AB = 0)
sqt1_ap_mid = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 1, AB = 1)
sqt1_ap_apex = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 1, AB = 2)

# Plot the AB gradient
plt.figure()
plt.plot(sqt1_ap_base['data']['engine.time'], sqt1_ap_base['data']['membrane.V'], label=f"Base APD90 = {sqt1_ap_base['duration']} ms")
plt.plot(sqt1_ap_mid['data']['engine.time'], sqt1_ap_mid['data']['membrane.V'], label=f"Mid APD90 = {sqt1_ap_mid['duration']} ms")
plt.plot(sqt1_ap_apex['data']['engine.time'], sqt1_ap_apex['data']['membrane.V'], label=f"Apex APD90 = {sqt1_ap_apex['duration']} ms")
plt.legend()
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('Membrane Potential')
plt.xlim([0, 300])
plt.tight_layout()

# Additional simulations of WT and SupRep
wt_ap = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 0, AB = 0)
sqt1_ap = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 1, AB = 0)
suprep_ap = action_pot(m = m, p = pace, bcl = bcl, prepace = 1000, cond = 2, AB = 0)

# Create a figure with two subplots (stacked vertically)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(wt_ap['data']['engine.time'], wt_ap['data']['membrane.V'], label=f"WT APD90 = {wt_ap['duration']} ms")
plt.plot(sqt1_ap['data']['engine.time'], sqt1_ap['data']['membrane.V'], label=f"SQT1 APD90 = {sqt1_ap['duration']} ms")
plt.plot(suprep_ap['data']['engine.time'], suprep_ap['data']['membrane.V'], label=f"SupRep APD90 = {suprep_ap['duration']} ms")
plt.legend()
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('Membrane Potential')
plt.xlim([0, 500])
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(wt_ap['data']['engine.time'], wt_ap['data']['ikr.IKr'], label="WT IKr")
plt.plot(sqt1_ap['data']['engine.time'], sqt1_ap['data']['ikr.IKr'], label="SQT1 IKr")
plt.plot(suprep_ap['data']['engine.time'], suprep_ap['data']['ikr.IKr'], label="SupRep IKr")
plt.legend()
plt.xlabel('Time [ms]')
plt.ylabel('IKr current [A]')
plt.title('IKr Current')
plt.xlim([0, 500])
plt.tight_layout()

#%% Export the AP data
wt_df = pd.DataFrame({
    'Time': wt_ap['data']['engine.time'],
    'Membrane Potential': wt_ap['data']['membrane.V'],
    'IKr': wt_ap['ikr']})

sqt1_df = pd.DataFrame({
    'Time': sqt1_ap['data']['engine.time'],
    'Membrane Potential': sqt1_ap['data']['membrane.V'],
    'IKr': sqt1_ap['ikr']})

suprep_df = pd.DataFrame({
    'Time': suprep_ap['data']['engine.time'],
    'Membrane Potential': suprep_ap['data']['membrane.V'],
    'IKr': suprep_ap['ikr']})

wt_df.to_csv('wt_df.csv', index = False)
sqt1_df.to_csv('sqt1_df.csv', index = False)
suprep_df.to_csv('suprep_df.csv', index = False)