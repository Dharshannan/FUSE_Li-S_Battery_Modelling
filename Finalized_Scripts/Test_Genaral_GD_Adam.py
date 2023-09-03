import pickle
import os
#import matplotlib.pyplot as plt
from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
from scipy.interpolate import splrep, splev
import joblib ## Used for parallel/efficient solver function runs 

# =============================================================================
# Helper Functions below for Gradient Descent Methodology
# =============================================================================
## Postive delta function
def input_delta_pos(input_dict, deltas):
    for key, value in deltas:
        if key in input_dict:
            input_dict[key] += value
        else:
            print("Warning: Deltas List and Dictionary of Parameters Do Not Match")
    return(input_dict)

## Negative delta function
def input_delta_neg(input_dict, deltas):
    for key, value in deltas:
        if key in input_dict:
            input_dict[key] -= value
        else:
            print("Warning: Deltas List and Dictionary of Parameters Do Not Match")
    return(input_dict)

# This function saturates the parameter values at its respective min and max values
def saturate_param(max_param, min_param, opt_param):
    # Turn lists to dictionaries
    max_param = dict(max_param)
    min_param = dict(min_param)
    ## Iterate through the param dict
    for key in opt_param and min_param:
        ## Saturate at minimum
        opt_param[key] = max(opt_param[key], min_param[key])
        
    for key in opt_param and max_param:
        ## Saturate at maximum
        if opt_param[key] > max_param[key]:
            opt_param[key] = max_param[key]
    ## Return
    return(opt_param)

## Function to find duplicate values that repeats
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

## Define ADAM function to return the optimized 1st and 2nd moments and step size values
def ADAM (grad, m_init, v_init, beta1, beta2, alpha_init, t):
    
    m_new = beta1*m_init + (1 - beta1)*grad
    v_new = beta2*v_init + (1 - beta2)*(grad**2)
    
    m_hat = m_new/(1 - beta1**t)
    v_hat = v_new/(1 - beta2**t)
    
    alpha_new = alpha_init*np.sqrt(1 - beta2**t)/(1 - beta1**t)
    
    ## Return a dictionary for easier handling
    return({'m_hat': m_hat, 'v_hat': v_hat, 'alpha_new': alpha_new, 'm_new': m_new, 'v_new': v_new})

# =============================================================================
# This is now a Generalized ADAM Gradient Descent Solver
# =============================================================================

## We define function to return simulated results and experimental data interpolated ##
## We can pass the arguments in terms of the new values of parameters for optimization
def ret_data(opt_params, backtrack):
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

    h_try = [0.5, 0.05, 0.005, 0.0005, 0.00005] # Step sizes to try
    tries = 0 # Number of tries executed

    while tries < len(h_try):

        try:
            ## Now we call the solver and solve ##
            t_end = 7500 # End time
            h0 = h_try[tries] # Initial step size
            
            ## Initialize the variable values and arrays
            ## We can change this to take in values based of dictionary key for each initial condition
            ## These are the actual values previously used
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
            
            # Uncomment these and comment out the above if initial values are to be optimized
# =============================================================================
#             Li_cath = opt_params["Li_cath"]
#             s8_cath = opt_params["s8_cath"]
#             s4_cath = opt_params["s4_cath"]
#             s2_cath = opt_params["s2_cath"]
#             s1_cath = opt_params["s1_cath"]
#             sp_cath = opt_params["sp_cath"]
#             Li_sep = opt_params["Li_sep"]
#             s8_sep = opt_params["s8_sep"]
#             s4_sep = opt_params["s4_sep"]
#             s2_sep = opt_params["s2_sep"]
#             s1_sep = opt_params["s1_sep"]
# =============================================================================
            
            V = 2.5279911819843837
            I = 2*0.211*0.2
            x_var = [Li_cath, s8_cath, s4_cath, s2_cath, s1_cath, sp_cath, Li_sep, s8_sep, s4_sep, s2_sep, s1_sep, V]
            
            break_voltage = opt_params['EL0'] - 0.1 # Voltage point where simulation is stopped to prevent Singular Matrix Error
            
            upd_param = opt_params
            ## Use the backtrack dict to update the params_backtracked
            params_backtracked = {}
            for key in backtrack:
                params_backtracked[key] = opt_params[key]*backtrack[key]
            
            ## Run the solver and save results within npz file
            solved = LiS_Solver(x_var, t_end, h0, I, break_voltage, state='Discharge', 
                                params_backtrack=params_backtracked, upd_params=upd_param)

            # Return Voltage and time arrays
            V = solved[-2]
            t = solved[-1]
            
            ## Turn all arrays to numpy arrays
            V = np.array(V)
            t = np.array(t)
            Ah = (t/3600)*I
            index_break2 = next((index for index, value in enumerate(Ah) if value > 0.15), None)
            
            break
            
        except Exception as e:
            if tries >= len(h_try) - 1:
                raise
                break

            tries += 1
    
    # =============================================================================
    # =============================================================================
    # Interpolate the experimental array to ensure that both arrays have the same range
    # =============================================================================
    # =============================================================================
    C_exp = scale_factor*discharge_data['30']['capacity'][:index_break]
    V_exp = discharge_data['30']['internal voltage'][:index_break]
    
    Ah = Ah[:index_break2]
    V = V[:index_break2]
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
def cost_fn(sim, exp, phi, gamma):
    error = sim - exp
    if error >= 0:
        return gamma * error
    else:
        return phi * abs(error)

# =============================================================================
# ## Now we define the gradient descent solver function
# =============================================================================
def Gradient_Descent_ADAM(init_vals, delta_params, epoch, beta1, beta2, alpha_param, max_param, 
                          min_param, backtracked):
    ## Change init_vals, alpha param and backtracked to dictionaries
    opt_params = dict(init_vals)
    alpha_param = dict(alpha_param)
    backtracked = dict(backtracked)
    
    ## These are used for the cost function
    phi = 2.5
    gamma = 2.25
    
    ## Initiate the ADAM 1st and 2nd moments using a dictionary
    Adam_params = {}
    for key, value in delta_params:
        new_dict = {"m": 0, "v": 0}
        Adam_params[key] = new_dict
    
    ## Start epoch iterations
    for i in range(epoch):
        ## Get the dictionaries to be passed into the LiS_Solver function for simulation
        dicts_to_sim = []
        for k in range(len(delta_params)):
            k_deltas = [delta_params[k]]
            
            neg_val = input_delta_neg(opt_params.copy(), k_deltas)
            pos_val = input_delta_pos(opt_params.copy(), k_deltas)
            ## This follows the deltas sequence of parameter delta updates
            dicts_to_sim.append([pos_val, neg_val])
            
        dicts_to_sim = [(item, backtracked) for sublist in dicts_to_sim for item in sublist]
        
        ## Now we initiate solver using Joblib and os cpu count
        ## This ensures the parallel computations are spread across available cpu cores
        max_operations = os.cpu_count()
        n_operations = len(delta_params)*2
        
        results = joblib.Parallel(n_jobs=min(n_operations,max_operations), backend='loky')(
            joblib.delayed(ret_data)(*params) for params in dicts_to_sim)
        
        ## Now append the results into a dictionary
        param_res = {}
        index = 0
        for key, value in delta_params:
            new_dict = {"pos": results[index], "neg": results[index + 1]}
            param_res[key] = new_dict
            index += 2
        
        ## Now calculate the gradients and append each into repective dictionaries
        grad_dict = {}
        indx = 0
        for key in param_res:
            pos_res = param_res[key]["pos"]
            neg_res = param_res[key]["neg"]
            pos_cost = 0
            neg_cost = 0
            
            for j in range(len(pos_res[0])):
                ## Set the values of phi and gamma correctly when implementing in actual function
                pos_cost += cost_fn(pos_res[1][j], pos_res[0][j], phi, gamma)
            
            for j in range(len(neg_res[0])):
                ## Set the values of phi and gamma correctly when implementing in actual function
                neg_cost += cost_fn(neg_res[1][j], neg_res[0][j], phi, gamma)
                
            delta_grad = delta_params[indx][1]
            grad_param = (pos_cost - neg_cost)/(2*delta_grad)
            grad_dict[key] = grad_param
            indx += 1
            
        ## Now we calculate and update new parameter values and update ADAM moments
        for key in Adam_params:
            get_ADAM = ADAM(grad_dict[key], Adam_params[key]["m"], Adam_params[key]["v"], 
                            beta1, beta2, alpha_param[key], i+1)
            
            m_hat = get_ADAM["m_hat"]
            v_hat = get_ADAM["v_hat"]
            alpha = get_ADAM["alpha_new"]
            ## Calculate next values using ADAM Gradient Descent
            eps = 1e-8
            opt_params[key] = opt_params[key] - alpha*m_hat/(np.sqrt(v_hat) + eps)
            ## Update the moments for the next iteration
            Adam_params[key]["m"] = get_ADAM["m_new"]
            Adam_params[key]["v"] = get_ADAM["v_new"]
            
        ## Saturate param values using saturate function
        print("=================================================================================")
        print("\u001b[1;32m" + f"Epoch: {i+1}/{epoch}" + "\u001b[0m")
        print("\033[0;37m" + f"Before Saturation: {opt_params}" + "\u001b[0m")
        opt_params = saturate_param(max_param, min_param, opt_params)
        print("After Saturation :", opt_params)
        print("=================================================================================")
        
    ## Print and Return the optimized parameter values as a dictionary
    print("=================================================================================")
    print("\033[1;36m" + f"Optimized Parameters: {opt_params}" + "\u001b[0m")
    print("=================================================================================")
    return(opt_params)

# =============================================================================
# Now we test this bad boi, HELL YEAH ;) !!!
# =============================================================================

# =============================================================================
# This part is if the initial values for the variables are to be optimized
# =============================================================================
# =============================================================================
# ## Initial variable values (Used as guess values)
# variables = [
#     ("Li_cath", 23.618929391226814),
#     ("s8_cath", 19.672721954609568),
#     ("s4_cath", 0.011563045206703666 * 1000),
#     ("s2_cath", 0.0001),
#     ("s1_cath", 4.886310174346254e-10),
#     ("sp_cath", 0.008672459420571042),
#     ("Li_sep", 41.96561647689176),
#     ("s8_sep", 19.433070078021764),
#     ("s4_sep", 18.597902007945958),
#     ("s2_sep", 0.0001),
#     ("s1_sep", 2.1218138582883716e-12)
# ]
# 
# var_delta = [
#     ("Li_cath", 0.01),
#     ("s8_cath", 0.01),
#     ("s4_cath", 0.01),
#     ("s2_cath", 0.000001),
#     ("s1_cath", 1e-12),
#     ("sp_cath", 0.00001),
#     ("Li_sep", 0.01),
#     ("s8_sep", 0.01),
#     ("s4_sep", 0.01),
#     ("s2_sep", 0.000001),
#     ("s1_sep", 1e-14)
# ]
# 
# var_alpha = [
#     ("Li_cath", 0.1),
#     ("s8_cath", 0.1),
#     ("s4_cath", 0.1),
#     ("s2_cath", 0.00001),
#     ("s1_cath", 1e-11),
#     ("sp_cath", 0.0001),
#     ("Li_sep", 0.1),
#     ("s8_sep", 0.1),
#     ("s4_sep", 0.1),
#     ("s2_sep", 0.00001),
#     ("s1_sep", 1e-13)
# ]
# 
# ## Initiate the variables and its dependent parameters and then call the Gradient Descent function
# init_params = [("EL0", 2.0), ("EM0", 2.3), ("kp", 500)]
# init_vals = init_params + variables
# 
# ## This (delta_params) is the step change for "+" and "-" used to calculate the derivative of cost function w.r.t parameter
# delta_params = [("EL0", 0.01), ("EM0", 0.01)] + var_delta
# 
# epoch = 100
# beta1 = 0.9
# beta2 = 0.999
# 
# ## This (alpha_param) is the initial learning rate for each parameter (Dynamically Updated)
# alpha_param = [("EL0", 0.1), ("EM0", 0.1), ("kp", 5)] + var_alpha
# 
# ## For now make sure max_param and min_param have the same keys (*Now it does not essentially require the same keys) 
# max_param = [("EL0", 2.1), ("EM0", 2.3), ("kp", 1000)]
# min_param = [("EL0", 1.9), ("EM0", 1.9), ("kp", 1)]
# 
# ## Parameter to backtrack and percentage value to backtrack (i.e: 0.5% increase = 1.005)
# backtracked = [("EL0", 1.005)]
# 
# ## Call gardient descent function and pass defined arguments 
# optimized = Gradient_Descent_ADAM(init_vals, delta_params, epoch, beta1, beta2,
#                                   alpha_param, max_param, min_param, backtracked)
# print(optimized)
# =============================================================================

# =============================================================================
# This part is if only parameter values are to be optimized
# =============================================================================
init_vals = [("EL0", 1.9), ("EM0", 1.9308455694919777), ("kp", 137.99786642485066)]
## This (delta_params) is the step change for "+" and "-" used to calculate the derivative of cost function w.r.t parameter
delta_params = [("EL0", 0.01), ("EM0", 0.01), ("kp", 1)]
epoch = 100
beta1 = 0.9
beta2 = 0.999
## This (alpha_param) is the initial learning rate for each parameter (Dynamically Updated)
alpha_param = [("EL0", 0.001), ("EM0", 0.0005), ("kp", 100)]
## For now make sure max_param and min_param have the same keys (*Now it does not essentially require the same keys) 
max_param = [("EL0", 2.1), ("EM0", 2.3), ("kp", 500)]
min_param = [("EL0", 1.9), ("EM0", 1.9), ("kp", 1)]
## Parameter to backtrack and percentage value to backtrack (i.e: 0.5% increase = 1.005)
backtracked = [("EL0", 1.005)]
## Call gardient descent function and pass defined arguments 
optimized = Gradient_Descent_ADAM(init_vals, delta_params, epoch, beta1, beta2,
                                  alpha_param, max_param, min_param, backtracked)
print(optimized)
