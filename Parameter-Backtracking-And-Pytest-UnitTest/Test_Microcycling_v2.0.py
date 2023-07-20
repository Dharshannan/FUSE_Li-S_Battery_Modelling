import numpy as np
from LiS_Backtrack_Solver import LiS_Solver, LiS_Solver2
import timeit

## Now we call the solver and solve ##
span = 3600
t0 = 0
t_end = t0 + span # End time
h0 = 0.5 # Initial step size

## Initialize the variable values and arrays for microcycling
## Start with Discharge 1st
s8i = 2.6892000000000003
s4i = 0.0027
s2i = 0.002697299116926997
si = 8.83072852310722e-10
spi = 2.7e-06
Vi = 2.430277479547109
I = 1.7 # Current (constant)
discharge_break = 1.985 # Voltage point where simulation is stopped to prevent Singular Matrix Error
charge_break = Vi

cycles = 5 ## Number of cycles to run (1 cycle is Discharge followed by Charge)
overall_array = []
for j in range(cycles):
    overall_array.append([])
    
## Define backtracking parameter values
params_backtracked = {"EL0": 2.0}

## Define different value of kp for charging
charge_upd = {"kp": 125}
start = timeit.default_timer() ## Start timer
## Run the solver within a for loop and cycle between Discharge and Charge
for i in range(cycles):
    # Discharge
    solved = LiS_Solver(s8i, s4i, s2i, si, Vi, spi, 
                        t_end, h0, I, discharge_break, state='Discharge', t0=t0, 
                        params_backtrack = params_backtracked)
    
    print(f'Cycle {i+1} Discharge Solved')
    
    list1 = solved
    overall_array[i].append(list1)
    s8i = solved[0][-1]
    s4i = solved[1][-1]
    s2i = solved[2][-1]
    si = solved[3][-1]
    Vi = solved[4][-1]
    spi = solved[5][-1]
    t0 = solved[6][-1]
    t_end = t0 + span

    # Charge
    solved2 = LiS_Solver2(s8i, s4i, s2i, si, Vi, spi, 
                        t_end, h0, -I, charge_break, state='Charge', t0=t0, 
                        params_backtrack = params_backtracked, upd_params = charge_upd)
    
    print(f'Cycle {i+1} Charge Solved')
    
    list2 = solved2
    overall_array[i].append(list2)
    s8i = solved2[0][-1]
    s4i = solved2[1][-1]
    s2i = solved2[2][-1]
    si = solved2[3][-1]
    Vi = solved2[4][-1]
    spi = solved2[5][-1]
    t0 = solved2[6][-1]
    t_end = t0 + span
      
    print(f'No. Cycles: {i+1}/{cycles}')
print("The time taken for completion :", timeit.default_timer() - start, "s")
overall_array_np = np.empty(len(overall_array), dtype=object)
overall_array_np[:] = overall_array

## Now we save the array above ##
## The results can be obtained in the Test_Saved_Microcycling.py script
np.savez('variable_arrays.npz', solved=overall_array_np, I=I)
print("Solved array returned in the form: [s8_array, s4_array, s2_array, s_array, V_array, sp_array, time_array]")
print("The indexing of the variables follows the list above, Ex: Voltage is index:4 or Precipitated Sulfur is index:5")