from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
import matplotlib.pyplot as plt

h_try = [1.25, 0.5, 0.05, 0.005] # Step sizes to try
tries = 0 # Number of tries executed

while tries < len(h_try):
    print("===========================================================")
    try:
        ## Now we call the solver and solve ##
        t_end = 11000 # End time
        h0 = h_try[tries] # Initial step size
        
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
        
        param_EL0 = 1.8
        break_voltage = param_EL0 # Voltage point where simulation is stopped to prevent Singular Matrix Error
        
        upd_param = {}
        params_backtracked = {"EL0": 1.94*1.005}
        
        ## Run the solver and save results within npz file
        solved = LiS_Solver(x_var, t_end, h0, I, break_voltage, state='Discharge', 
                            params_backtrack=params_backtracked, upd_params=upd_param)

        V = solved[-2]
        t = solved[-1]
        Li = solved[0]
        
        ## Turn all arrays to numpy arrays
        V = np.array(V)
        Li = np.array(Li)

        # Now we plot:
        t = np.array(t)
        Ah = (t/3600)*I
        plt.plot(Ah,V)
        plt.title(f"Cell Output Voltage vs Discharge Capacity {round(I,4)}A")
        plt.ylabel("Potential (V)")
        plt.xlabel("Discharge Capacity (Ah)")
        #plt.xlim([-0.0002, 0.18])
        min_y = break_voltage - 0.05
        #plt.ylim([min_y, 2.45])
        plt.savefig('Voltage_Output_40C.png', dpi=1200)
        plt.show()
        
        plt.plot(t, Li)
        plt.show()
        
        print(f"Successful Run step size: {h_try[tries]}")
        print("===========================================================")
        break
        
    except Exception as e:
        if tries >= len(h_try) - 1:
            print("No Successful Runs, Simulation Terminated!")
            print("===========================================================")
            raise
            break
        print(f"Error occurred: {e}")
        print(f"Changing step size from {h_try[tries]} to {h_try[tries+1]}")
        print("===========================================================")
        print("\n")
        tries += 1