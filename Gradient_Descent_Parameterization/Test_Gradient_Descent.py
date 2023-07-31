import pickle
import matplotlib.pyplot as plt
from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
from scipy.interpolate import splrep, splev

# =============================================================================
# ## Carry out some data-processing: ##
# =============================================================================
infile = open('vol_data', 'rb')
data_dict = pickle.load(infile)
infile.close()

discharge_data = data_dict['discharge']
data_array = 20*discharge_data['30']['capacity']
index_break = next((index for index, value in enumerate(data_array) if value > 3.24), None)

## Now we call the solver and solve ##
t_end = 7500 # End time
h0 = 0.0005 # Initial step size

## Initialize the variable values and arrays
s8i = 2.6892000000000003
s4i = 0.0027
s2i = 0.002697299116926997
si = 8.83072852310722e-10
spi = 2.7e-06
Vi = 2.430277479547109
I = 1.7 # Current (constant)
EL0 = 1.9883238535375416
kp = 0.6286029102986248
break_voltage = EL0 # Voltage point where simulation is stopped to prevent Singular Matrix Error

upd_param = {"EL0": EL0, "kp": kp}
params_backtracked = {"EL0": EL0*1.005}

## Run the solver and save results within npz file
solved = LiS_Solver(s8i, s4i, s2i, si, Vi, spi, t_end, h0, I, break_voltage, state='Discharge', 
                    params_backtrack=params_backtracked, upd_params=upd_param)
#print(solved)
#np.savez('variable_arrays.npz', solved=solved, I=I)
s8 = solved[0]
s4 = solved[1]
s2 = solved[2]
s = solved[3]
V = solved[4]
sp = solved[5]
t = solved[6]

## Turn all arrays to numpy arrays
s8 = np.array(s8)
s4 = np.array(s4)
s2 = np.array(s2)
s = np.array(s)
V = np.array(V)
sp = np.array(sp)
t = np.array(t)
Ah = (t/3600)*I

# Now we plot:
# Plot simulation
plt.plot(Ah,V, label="Simulated Results")

# Plot experimental
plt.plot(20*discharge_data['30']['capacity'][:index_break], 
         discharge_data['30']['internal voltage'][:index_break], label="Experimental Data")
plt.title("Cell Output Voltage vs Discharge Capacity")
plt.ylabel("Potential (V)")
plt.xlabel("Discharge Capacity (Ah)")
plt.legend(framealpha=1, frameon=True)
#plt.savefig('1.7A_Voltage_2.png', dpi=1200)
plt.show()

# =============================================================================
# 
# Interpolate the experimental array to ensure that both arrays have the same values
# 
# =============================================================================
# =============================================================================
# C_exp = 20*discharge_data['30']['capacity'][:index_break]
# V_exp = discharge_data['30']['internal voltage'][:index_break]
# 
# C_sim = Ah
# V_sim = V
# 
# tck_exp = splrep(C_exp, V_exp, s=0)
# tck_sim = splrep(C_sim, V_sim, s=0)
# 
# C_common = np.arange(0, Ah[-1], Ah[-1]/1000)
# 
# V_exp_interpolated = splev(C_common, tck_exp, der=0)
# V_sim_interpolated = splev(C_common, tck_sim, der=0)
# 
# plt.plot(C_common, V_exp_interpolated, label="Experimental Data")
# plt.plot(C_common, V_sim_interpolated, label="Simulated Results")
# plt.title("Cell Output Voltage vs Discharge Capacity")
# plt.ylabel("Potential (V)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.legend(framealpha=1, frameon=True)
# plt.show()
# =============================================================================
