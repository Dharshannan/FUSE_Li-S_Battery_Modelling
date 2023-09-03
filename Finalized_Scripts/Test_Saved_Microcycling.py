import numpy as np
import module_func
import matplotlib.pyplot as plt

## Load Saved Data ##
data = np.load('variable_arrays.npz', allow_pickle=True)
overall_array_np = data['solved']

labels = module_func.labels(overall_array_np) ## Create the dictionary
## Access cycle1 discharge ##
cyc = "cycle1" ## Variable to access cycle
state = "discharge" ## Variable to access state
cycle = labels[cyc][state] 
cyc_t = cycle[-1]/3600 ## Time array stored in last index
cyc_V = cycle[-2] ## Voltage array stored in 2md to last index

## Call concatenate function if whole microcycling process is wanted
## The 2nd argument in this function is the index of the variable wanted
whole_t = module_func.concatenate(labels, -1)/3600
whole_V = module_func.concatenate(labels, -2)
whole_Li = module_func.concatenate(labels, 0)

## Plot Data for whole microcycling
plt.plot(whole_t, whole_V)
plt.title('Microcycling Plot of Voltage (Discharge and Charge) vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
#plt.savefig('Microcycling_Voltage.png', dpi=1200)
plt.show()

plt.plot(whole_t, whole_Li)
plt.title('Microcycling Plot of Li Cathode (Discharge and Charge) vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Li Cathode (mol/L)')
#plt.savefig('Microcycling_Voltage.png', dpi=1200)
plt.show()

## Plot Data for each cycle
plt.plot(cyc_t, cyc_V)
plt.title(f"Plot of {cyc}, state:{state}, V vs time")
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.show()

