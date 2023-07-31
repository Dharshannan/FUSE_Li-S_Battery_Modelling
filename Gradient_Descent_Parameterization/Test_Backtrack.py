from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
import matplotlib.pyplot as plt

h_try = [1, 0.5, 0.05] # Step sizes to try
tries = 0 # Number of tries executed

while tries < len(h_try):
    print("===========================================================")
    try:
        ## Now we call the solver and solve ##
        t_end = 7500 # End time
        h0 = h_try[tries] # Initial step size
        
        ## Initialize the variable values and arrays
        s8i = 2.6892000000000003
        s4i = 0.0027
        s2i = 0.002697299116926997
        si = 8.83072852310722e-10
        spi = 2.7e-06
        Vi = 2.430277479547109
        I = 1.7 # Current (constant)
        
        param_EL0 = 1.975
        break_voltage = param_EL0 # Voltage point where simulation is stopped to prevent Singular Matrix Error
        
        upd_param = {"EL0": param_EL0, "kp": 3.5678}
        params_backtracked = {"EL0": param_EL0*1.005}
        
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
        ## Calculate partial current iH and iL values for plotting
        ar = 0.96
        fh = 0.73
        fl = 0.067
        EH0 = 2.4
        EL0= 1.95
        iH0 = 10
        iL0 = 5
        R = 8.3145
        T = 298
        F = 96485 
        ne = 4
        
        EH = EH0 + ((R*T)/(4*F))*np.log(fh*s8/(s4**2))
        EL = EL0 + ((R*T)/(4*F))*np.log(fl*s4/(s2*(s**2)))
        
        nH = V - EH
        nL = V - EL
        
        iH = -2*iH0*ar*np.sinh((ne*F*nH)/(2*R*T))
        iL = -2*iL0*ar*np.sinh((ne*F*nL)/(2*R*T))
        I_new = iH + iL
        # Now we plot:
        t = np.array(t)
        Ah = (t/3600)*I
        plt.plot(Ah,V)
        plt.title(f"Cell Output Voltage vs Discharge Capacity {I}A")
        plt.ylabel("Potential (V)")
        plt.xlabel("Discharge Capacity (Ah)")
        plt.xlim([-0.25,3.5])
        min_y = break_voltage - 0.05
        plt.ylim([min_y, 2.45])
        #plt.savefig('1.7A_Voltage_Output_New.png', dpi=1200)
        plt.show()
        
        # =============================================================================
        # plt.plot(Ah,iH)
        # plt.ylabel("iH (A)")
        # plt.xlabel("Discharge Capacity (Ah)")
        # plt.ylim([-0.5,2])
        # plt.show()
        # =============================================================================
        
        # =============================================================================
        # plt.plot(Ah, sp, c= 'orange')
        # plt.title("Precipitated Sulfur Mass vs Discharge Capacity")
        # plt.ylabel("Precipitated S (g)")
        # plt.xlabel("Discharge Capacity (Ah)")
        # plt.xlim([0.85,1.15])
        # plt.ylim([0,0.15])
        # plt.show()
        # =============================================================================
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