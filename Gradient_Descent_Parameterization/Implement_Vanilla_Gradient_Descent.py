import pickle
import matplotlib.pyplot as plt
from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
from scipy.interpolate import splrep, splev
import joblib ## Used for parallel/efficient solver function runs 

## Function to find duploicate values that repeats
def find_duplicates(input_list):
    seen_tuples = {}
    duplicates = []

    for tuple_item in input_list:
        if tuple_item in seen_tuples:
            if seen_tuples[tuple_item] == 1:
                duplicates.append(tuple_item)
            seen_tuples[tuple_item] += 1
        else:
            seen_tuples[tuple_item] = 1

    return duplicates

# =============================================================================
# Note: We will test a simple vanilla gradient descent with 2 parameters
# which will be kp and EL0 1st, and then try to implement this further.
# =============================================================================

## We define function to return simulated results and experimental data interpolated ##
## We can pass the arguments in terms of the new values of parameters for optimization
## Note: For Now lets just try with kp and EL0
def ret_data(kp, EL0):
    # =============================================================================
    # ## Carry out some data-processing: ##
    # =============================================================================
    infile = open('vol_data', 'rb')
    data_dict = pickle.load(infile)
    infile.close()

    discharge_data = data_dict['discharge']
    data_array = 20*discharge_data['30']['capacity']
    index_break = next((index for index, value in enumerate(data_array) if value > 3.0), None)

    h_try = [0.05, 0.005, 0.0005, 0.00005] # Step sizes to try
    tries = 0 # Number of tries executed

    while tries < len(h_try):

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
            
            break
            
        except Exception as e:
            if tries >= len(h_try) - 1:
                raise
                break

            tries += 1
    
    # =============================================================================
    # =============================================================================
    # Interpolate the experimental array to ensure that both arrays have the same values
    # =============================================================================
    # =============================================================================
    C_exp = 20*discharge_data['30']['capacity'][:index_break]
    V_exp = discharge_data['30']['internal voltage'][:index_break]

    C_sim = Ah
    V_sim = V

    tck_exp = splrep(C_exp, V_exp, s=0)
    tck_sim = splrep(C_sim, V_sim, s=0)

    C_common = np.arange(0, Ah[-1], Ah[-1]/1000)

    V_exp_interpolated = splev(C_common, tck_exp, der=0)
    V_sim_interpolated = splev(C_common, tck_sim, der=0)
    
    return([V_exp_interpolated, V_sim_interpolated])

# =============================================================================
# We will now define our cost function
# =============================================================================
# This is an asymmetric cost function
def cost_fn(sim, exp, alpha, beta):
    error = sim - exp
    if error >= 0:
        return beta * error
    else:
        return alpha * abs(error)

# =============================================================================
# ## Now we define a the gradient descent solver function
# =============================================================================
def gradient_descent(init_kp, init_EL0, delta_kp, delta_EL0, learning_rate, epoch):
    alpha = 1
    beta = 0.75
    repeat = []
    # Initiate parameters
    kp = init_kp
    EL0 = init_EL0
    # Start epoch iterations
    for i in range(epoch):
        # =============================================================================
        ## calculate cost for kp + delta_kp
        # =============================================================================
        ## Define array containing all new parameter values
        params_sub = [(kp + delta_kp, EL0), (kp - delta_kp, EL0), (kp, EL0 + delta_EL0), (kp, EL0 - delta_EL0)]
        # Run the simulation in parallel using joblib (improves run time significantly)
        results = joblib.Parallel(n_jobs=4, backend="loky")(joblib.delayed(ret_data)(*params) for params in params_sub)

        ## Now use the results from parallel simulations
        # =============================================================================
        ## calculate cost for kp + delta_kp
        # =============================================================================
        kp_data1 = results[0]
        pos_cost_kp = 0
        for j in range(len(kp_data1[0])):
            pos_cost_kp += cost_fn(kp_data1[1][j], kp_data1[0][j], alpha, beta)
        
        # =============================================================================
        ## calculate cost for kp - delta_kp
        # =============================================================================
        kp_data2 = results[1]
        neg_cost_kp = 0
        for j in range(len(kp_data2[0])):
            neg_cost_kp += cost_fn(kp_data2[1][j], kp_data2[0][j], alpha, beta)
            
        # =============================================================================
        ## calculate cost for EL0 + delta_EL0
        # =============================================================================
        EL0_data1 = results[2]
        pos_cost_EL0 = 0
        for j in range(len(EL0_data1[0])):
            pos_cost_EL0 += cost_fn(EL0_data1[1][j], EL0_data1[0][j], alpha, beta)
            
        # =============================================================================
        ## calculate cost for EL0 - delta_EL0
        # =============================================================================
        EL0_data2 = results[3]
        neg_cost_EL0 = 0
        for j in range(len(EL0_data2[0])):
            neg_cost_EL0 += cost_fn(EL0_data2[1][j], EL0_data2[0][j], alpha, beta)
        
        ## Now we carry out gradient descent for both the parameters seperately and update them
        EL0_next = EL0 - 0.1*learning_rate*(pos_cost_EL0 - neg_cost_EL0)/(2*delta_EL0)
        kp_next = kp - 5e2*learning_rate*(pos_cost_kp - neg_cost_kp)/(2*delta_kp)
        print("Actual kp, EL0 from GD:", kp_next, EL0_next)
        ## Set kp_next and EL0_next to be the new kp and EL0 values
        kp = max(kp_next, 1) ## Saturate kp at 5 if kp<5
        EL0 = max(EL0_next, 1.95) ## Saturate EL0 at 1.95 if EL0<1.95
        if EL0 > 2.0:
            EL0 = 2.0
        if kp > 10:
            kp = 10
        print(f"Epoch:{i+1}")
        print(f"kp:{kp}, EL0:{EL0}")
        repeat.append(tuple([kp, EL0]))
    ## Return kp and EL0 after epoch iteration is complete
    return([kp, EL0, repeat])

# =============================================================================
# ## Lets test run this bad boi!! Hell yeah!! ##
# =============================================================================

optimized = gradient_descent(1, 2.0, 0.0005, 0.0001, 1e-4, 100) ## Call optimization
repeated = find_duplicates(optimized[2]) ## Get duplicate values from potential bouncing
print("Repeated kp and EL0 from GD ,(kp, EL0):")
print(repeated)
