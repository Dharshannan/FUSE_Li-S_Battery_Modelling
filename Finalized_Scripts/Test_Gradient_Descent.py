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

scale_factor = 1
discharge_data = data_dict['discharge']
data_array = scale_factor*discharge_data['30']['capacity']
index_break = next((index for index, value in enumerate(data_array) if value > 0.15), None)

## Now we call the solver and solve ##
t_end = 7500 # End time # 7500
h0 = 1 # Initial step size

## Initialize the variable values and arrays
Li_cath = 23.618929391226814
s8_cath = 19.672721954609568
s4_cath = 0.011563045206703666*1000
s2_cath = 0.0001
s1_cath = 4.886310174346254e-10
sp_cath = 0.008672459420571042
Li_sep = 41.96561647689176
s8_sep = 19.433070078021764
s4_sep = 18.597902007945958
s2_sep = 0.0001
s1_sep = 2.1218138582883716e-12
V = 2.5279911819843837
I = 2*0.211*0.2
x_var = [Li_cath, s8_cath, s4_cath, s2_cath, s1_cath, sp_cath, Li_sep, s8_sep, s4_sep, s2_sep, s1_sep, V]

EL0 = 1.96
EM0 = 1.97
kp = 10
break_voltage = EL0 - 0.1 # Voltage point where simulation is stopped to prevent Singular Matrix Error

upd_param = {"EL0": EL0, "EM0": EM0, "kp": kp}
params_backtracked = {"EL0": EL0*1.005}

## Run the solver and save results within npz file
solved = LiS_Solver(x_var, t_end, h0, I, break_voltage, state='Discharge', 
                    params_backtrack=params_backtracked, upd_params=upd_param)

# Retunr voltage and time arrays
V = solved[-2]
t = solved[-1]

## Turn all arrays to numpy arrays
V = np.array(V)
t = np.array(t)
Ah = (t/3600)*I
index_break2 = next((index for index, value in enumerate(Ah) if value > 0.15), None)

# Now we plot:
# Plot simulation
Ah = Ah[:index_break2]
V = V[:index_break2]
plt.plot(Ah,V, label="Simulated Results")

# Plot experimental
plt.plot(scale_factor*discharge_data['30']['capacity'][:index_break], 
         discharge_data['30']['internal voltage'][:index_break], label="Experimental Data")
plt.title("Cell Output Voltage vs Discharge Capacity")
plt.ylabel("Potential (V)")
plt.xlabel("Discharge Capacity (Ah)")
plt.legend(framealpha=1, frameon=True)
#plt.savefig('1.7A_Voltage_1.png', dpi=1200)
plt.show()

# =============================================================================
# 
# Interpolate the experimental array to ensure that both arrays have the same values
# 
# =============================================================================

# =============================================================================
# C_exp = scale_factor*discharge_data['30']['capacity'][:index_break]
# V_exp = discharge_data['30']['internal voltage'][:index_break]
# 
# Ah = Ah[:index_break2]
# V = V[:index_break2]
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
