import numpy as np
#import timeit
import warnings
import func

class LiSModel:
    
    def __init__(self, x, I):
        # Define constants
        self.F = 96485.3321233100184 
        self.Ms = 32
        self.nH = 4
        self.nM = 2
        self.nL = 2
        self.ns8 = 8
        self.R = 8.3145
        self.ps = 2e3
        self.a = 0.96
        self.v = 1.14e-5
        self.EH0 = 2.35
        self.EM0 = 1.95
        self.EL0= 1.94
        self.jH0 = 1e-3
        self.jM0 = 1e-3
        self.jL0 = 1e-3
        self.CT0 = 165.51693435356822 # GD
        self.D8 = 0.01 * 0.75
        self.D4 = 0.000250 * 0.75
        self.D2 = 0.0000001 * 0.75
        self.D1 = 0.0000001 * 0.75
        self.DLi = 2.2625e-3 * 0.585 # GD
        self.kp = 0.45 # GD
        self.ks = 0
        self.Ksp = 1
        self.T = 292.15
        
        # Store variables
        self.x = x
        self.I = I
    
    def get_func_vals(self):
        # =============================================================================
        # Reference the u array and jacobian from the func.py script 
        # =============================================================================
        
        ## Reference the u_list
        self.u_list = func.u_list
        
        ## Reference the jacob_list
        self.jacob_list = func.jacob_list
    
    ## Now we define f function to get the u1-u6 function values and return in an array

    def f(self, h, prev_var):
        # Get list of parameters for substitution into lambified functions
        paramlist = func.param_list
        prefixed_list = [vars(self)[str(item)] for item in paramlist]

        ## Call get_func_vals() to get the values from the func.py script ##
        self.get_func_vals()
        x = self.x
        h_I = [h, self.I]

        x_new = list(x) + list(prev_var) + h_I + prefixed_list
        
        u_array = np.zeros(len(self.u_list))
        ## Now carry out substitution with values of variables
        for i in range(len(self.u_list)):
            u_array[i] = float(self.u_list[i](*x_new))
            
        # Return the u_array
        u_array = np.asarray(u_array, dtype='float64')
        return(u_array)
    
    ## Now define function to return Jacobian Matrix
    def jacobian(self, h):
        # Get list of parameters for substitution into lambified functions
        paramlist = func.param_list
        prefixed_list = [vars(self)[str(item)] for item in paramlist]

        ## Call get_func_vals() to get the values from the func.py script ##
        self.get_func_vals()
        x = self.x
        h_I = [h, self.I]

        x_new = list(x) + list(x) + h_I + prefixed_list
        
        jacob = np.zeros((len(self.jacob_list), len(self.jacob_list[0])))
        ## Substitute the values into the derivative functions
        for i in range(len(self.jacob_list)):
            for j in range(len(self.jacob_list[i])):
                ## The Jacobian is always a 2D square Matrix
                jacob[i][j] = float(self.jacob_list[i][j](*x_new))
                
        # Return the jacob array
        jacob = np.asarray(jacob, dtype='float64')
        return(jacob)
    
    ## Define function to update parameter/constant values versatilely 
    def update_parameters(self, **kwargs):
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

# =============================================================================  
# =============================================================================
# =============================================================================
#                Now we define the solver for the model class  
# =============================================================================
# =============================================================================
# =============================================================================

## x_var is a list of list ex: [[s8], [s4], ...[sp]]
def LiS_Solver(x_var, # This x_var variable will be a list containing all the variable values  
               t_end, h0, I, break_voltage, state = None, t0 = 0, backtracked = None, 
               params_backtrack = {}, upd_params = {}):
    
    ## Define state argument to handle breakpoints for charge and discharge seperately
    if state == None:
        state = 'Discharge' # Default to Discharge
    
    if state != None and state != 'Discharge' and state != 'Charge':
        raise ValueError('state can only be None, Discharge or Charge')
        
    ## The i above stands for initial, so s8i stands for initial s8 value ##
    t0 = t0
    t_end = t_end
    h0 = h0

    ## Initialize the variable values and arrays
    x_var = np.asarray(x_var) # convert to numpy array
    x_var = x_var[:, np.newaxis] # Append values to new axis to create 2D array
    t = [t0]
    h_step = [h0]
    jacob_array = []

    b = 0
    breakpoint1 = 0
    min_h = 1e-4 # Minimum step size
    max_h = 1.25 # Maximum step size
    max_jacobian = 0 # Variable to store maximum jacobian determinant
    
    # Set the warning filter to raise RuntimeWarning as an exception
    warnings.simplefilter("error", category=RuntimeWarning)
    
    ## Now start time iteration loop
    i = 1
    #start = timeit.default_timer() ## Start timer (Can be used in other scripts instead when calling the solver function)
    while t[-1] < t_end:
        
        try:
            # Initialize with guess values for each variable (use previous values as initial guesses)
            x_varguess = x_var[:, i-1]
            h = h_step[i-1]
            t_current = t[i-1]
    
            # Define the damping to be 1 at the start, we will dynamically update this if necessary
            lamda = 1.0 # Damping Factor
            damping_update_factor = 0.25
            damping_min = 1e-8
            regularization_factor = 5e-4
            
            while True:
                # Now calculate u (function) values and define jacobian elements using model class
                
                # Create a list to store the old guess values
                x = x_varguess
                
                ## Call model class and get the function arrays and jacobian
                model = LiSModel(x, I) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model.update_parameters(**upd_params)
                ## Change parameter values if recursion has happenend (Backtrack to set value)
                if backtracked == True:
                    model.update_parameters(**params_backtrack)
                ## Now solve as usual with new backtracked parameter value
                u_array = model.f(h, x_var[:, -1])
                jacob = model.jacobian(h)
                
                ### Now we calculate the determinant of the Jacobian and check to alter step size ###
                norm_jacob = np.linalg.det(jacob)
                max_ratio = 0

                # =============================================================================
                #     ### This will dynamically update the step size every iteration ###
                # =============================================================================
                if i > 1: ## Only check after 1st iteration ##
                    ## Continuously update the maximum value of determinant of Jacobian encountered
                    max_jacobian = max(max_jacobian, abs(jacob_array[i-3]))
                    ## Calculate the ratio of current determinant to maximum for Jacobian
                    max_ratio = abs(jacob_array[i-2])/max_jacobian
                    ## These values need configuration
                    if max_ratio >= 1.2: ## This indicates step needs to be reduced and parameter backtracking
                        ## Define recursive function call for parameter backtracking
                        ## Recursive call to only solve for 1 iteration:
                        t02 = 0
                        t_end2 = t02 + h
                        new_guess = LiS_Solver(x_varguess, 
                                               t_end2, h, I, break_voltage, state=state, t0=t02, backtracked=True, 
                                               params_backtrack=params_backtrack, upd_params=upd_params)
                        
                        ## Now update the guess values and run solver by updating u_array and jacobian
                        x_upd = np.asarray(new_guess[:-1, -1])
                        # Update the model
                        model = LiSModel(x_upd, I)
                        ## Change any values if a new value wants to be used apart from the default ones in the model class
                        model.update_parameters(**upd_params)
                        u_array = model.f(h, x_var[:, -1])
                        jacob = model.jacobian(h)
                        x = x_upd ## Use updated guess values from backtracking
                        
                        # =============================================================================
                        # Define Line Search Method to further optimize the step size           
                        # =============================================================================
                        jacobinv2 = np.linalg.inv(jacob)
                        delta2 = - np.matmul(jacobinv2,u_array)
                        alpha = 0.3
                        beta = 0.05
                        var_val = x.copy()
                        unew_array = u_array.copy()
        
                        while np.linalg.norm(unew_array) > np.linalg.norm(u_array + alpha*h*np.dot(jacob, delta2)):
                            upd_x = var_val + h*delta2
                            upd_x = abs(upd_x)
                            model3 = LiSModel(upd_x, I) ## Initialize the model
                            ## Change any values if a new value wants to be used apart from the default ones in the model class
                            model3.update_parameters(**upd_params)
                            unew_array = model3.f(h, x_var[:, -1])
                            h *= beta
                            if h < min_h:
                                break
                            #print(h, I*t[i-1]/3600)
                        
                        # Further Update h (step-size)
                        h_new = max(h*(0.25), min_h) ## Saturate at minimum step size
                        
                    elif max_ratio <= 1.0: ## This indicates step size can be increased 
                        h_new = min(h/(0.75), max_h) ## Saturate at maximum step size
                        
                    else:
                        h_new = h
                        
                else:
                    h_new = h
                
                ## Now we solve as usual the new step size will be implemented in the next iteration
                jacob = jacob + regularization_factor * np.eye(len(x))
                jacobinv = np.linalg.inv(jacob)
                # Calculate new values via Newton-Raphson
                delta = - np.matmul(jacobinv,u_array)
                new_val = x + lamda*delta
                new_val = abs(new_val)
                
                model2 = LiSModel(new_val, I) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model2.update_parameters(**upd_params)
                unew_array = model2.f(h, x_var[:, -1])
    
                # Compute the ratio of actual reduction to predicted reduction
                actual_reduction = np.linalg.norm(u_array) - np.linalg.norm(unew_array)
                predicted_reduction = np.linalg.norm(u_array) - np.linalg.norm(u_array + jacob @ delta)
                ratio = abs(actual_reduction / predicted_reduction)
    
                # Update the damping factor based on the ratio
                n_damp = 1
                if ratio > 1e-3:
                    lamda *= (damping_update_factor**n_damp)
                elif ratio < 1e-4:
                    lamda /= (damping_update_factor**n_damp)
                
                # Ensure the damping factor does not go below the minimum value
                lamda = max(lamda, damping_min)
    
                # Calculate error between new_val and old_guess
                err_list = np.zeros((len(x),))
                epsilon = 1e-10
                for j in range(len(err_list)):
                    err_list[j] = abs((new_val[j] - x[j])/(x[j] + epsilon))
                    
                # Check if all absolute difference/err is smaller than a specified tolerance
                tol = 1e-2
                ## Get the V_index (Voltage index from func.py)
                V_index = func.V_index
                
                if np.all(err_list<tol):
                    # If the error is small values have converged hence, update the new variables and break the while loop
                    t_next = t_current + h_new
                    t.append(t_next)
                    h_step.append(h_new)
                    jacob_array.append(norm_jacob)
                    
                    ## Voltage saturation ##
                    if state == 'Discharge':
                        new_val[V_index] = max(new_val[V_index], break_voltage)
                    elif state == 'Charge':
                        new_val[V_index] = min(new_val[V_index], break_voltage)
                        
                    ## Concatenate the new values to the exsisting list
                    x_var = np.concatenate((x_var, new_val[:, np.newaxis]), axis=1)
                    break
    
                else:
                    # Update guess values to be new values
                    ## Voltage saturation ##
                    if state == 'Discharge':
                        new_val[V_index] = max(new_val[V_index], break_voltage)
                    elif state == 'Charge':
                        new_val[V_index] = min(new_val[V_index], break_voltage)
                        
                    x_varguess = new_val 
    
            b = b + 1
            i = i + 1
    
            #print(f"No of iterations: {b}/{int(t_end/h)}")
            
            ### Handle breakpoint for discharge state ###
            ## Get the V_index (Voltage index from func.py)
            V_index = func.V_index
            V_break = x_var[V_index][i-1]
            
            if (V_break<=break_voltage) and (state == 'Discharge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var[:, :breakpoint1]
                t_array = np.asarray(t[:breakpoint1])
            
                return(np.vstack((var_array, t_array)))
                break
            
            ### Handle breakpoint for charge state ###
            elif (V_break>=break_voltage) and (state == 'Charge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var[:, :breakpoint1]
                t_array = np.asarray(t[:breakpoint1])
            
                return(np.vstack((var_array, t_array)))
                break
            
            elif (t[-1] >= t_end):
    # =============================================================================
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var
                t_array = np.asarray(t)
            
                return(np.vstack((var_array, t_array)))
                break
        
        ## When a warning is encountered break the loop: ##
        except RuntimeWarning as warning:
            print("Runtime Warning Encountered at Capacity:", t[i-1]/3600)
            warnings.warn(warning)  # Raise the RuntimeWarning again
            break

# =============================================================================  
# =============================================================================
# =============================================================================
#                Now we define the solver2 for the model class  
# =============================================================================
# =============================================================================
# =============================================================================

## x_var is a list of list ex: [[s8], [s4], ...[sp]]
def LiS_Solver2(x_var, # This x_var variable will be a list containing all the variable values  
               t_end, h0, I, break_voltage, state = None, t0 = 0, backtracked = None, 
               params_backtrack = {}, upd_params = {}):
    
    ## Define state argument to handle breakpoints for charge and discharge seperately
    if state == None:
        state = 'Discharge' # Default to Discharge
    
    if state != None and state != 'Discharge' and state != 'Charge':
        raise ValueError('state can only be None, Discharge or Charge')
        
    ## The i above stands for initial, so s8i stands for initial s8 value ##
    t0 = t0
    t_end = t_end
    h0 = h0

    ## Initialize the variable values and arrays
    x_var = np.asarray(x_var) # convert to numpy array
    x_var = x_var[:, np.newaxis] # Append values to new axis to create 2D array
    t = [t0]
    h_step = [h0]
    jacob_array = []

    b = 0
    breakpoint1 = 0
    min_h = 1e-6 # Minimum step size
    max_h = 1.25 # Maximum step size
    max_jacobian = 0 # Variable to store maximum jacobian determinant
    
    # Set the warning filter to raise RuntimeWarning as an exception
    warnings.simplefilter("error", category=RuntimeWarning)
    
    ## Now start time iteration loop
    i = 1
    #start = timeit.default_timer() ## Start timer (Can be used in other scripts instead when calling the solver function)
    while t[-1] < t_end:
        
        try:
            # Initialize with guess values for each variable (use previous values as initial guesses)
            x_varguess = x_var[:, i-1]
            h = h_step[i-1]
            t_current = t[i-1]
    
            # Define the damping to be 1 at the start, we will dynamically update this if necessary
            lamda = 1.0 # Damping Factor
            damping_update_factor = 0.25
            damping_min = 1e-8
            regularization_factor = 5e-4
            
            while True:
                # Now calculate u (function) values and define jacobian elements using model class
                
                # Create a list to store the old guess values
                x = x_varguess
                
                ## Call model class and get the function arrays and jacobian
                model = LiSModel(x, I) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model.update_parameters(**upd_params)
                ## Change parameter values if recursion has happenend (Backtrack to set value)
                if backtracked == True:
                    model.update_parameters(**params_backtrack)
                ## Now solve as usual with new backtracked parameter value
                u_array = model.f(h, x_var[:, -1])
                jacob = model.jacobian(h)
                
                ### Now we calculate the determinant of the Jacobian and check to alter step size ###
                norm_jacob = np.linalg.det(jacob)
                max_ratio = 0

                # =============================================================================
                #     ### This will dynamically update the step size every iteration ###
                # =============================================================================
                if i > 1: ## Only check after 1st iteration ##
                    ## Continuously update the maximum value of determinant of Jacobian encountered
                    max_jacobian = max(max_jacobian, abs(jacob_array[i-3]))
                    ## Calculate the ratio of current determinant to maximum for Jacobian
                    max_ratio = abs(jacob_array[i-2])/max_jacobian
                    ## These values need configuration
                    if max_ratio >= 1.2: ## This indicates step needs to be reduced and parameter backtracking
                        ## Define recursive function call for parameter backtracking
                        ## Recursive call to only solve for 1 iteration:
                        t02 = 0
                        t_end2 = t02 + h
                        new_guess = LiS_Solver(x_varguess, 
                                               t_end2, h, I, break_voltage, state=state, t0=t02, backtracked=True, 
                                               params_backtrack=params_backtrack, upd_params=upd_params)
                        
                        ## Now update the guess values and run solver by updating u_array and jacobian
                        x_upd = np.asarray(new_guess[:-1, -1])
                        # Update the model
                        model = LiSModel(x_upd, I)
                        ## Change any values if a new value wants to be used apart from the default ones in the model class
                        model.update_parameters(**upd_params)
                        u_array = model.f(h, x_var[:, -1])
                        jacob = model.jacobian(h)
                        x = x_upd ## Use updated guess values from backtracking
                        
                        # =============================================================================
                        # Define Line Search Method to further optimize the step size           
                        # =============================================================================
                        jacobinv2 = np.linalg.inv(jacob)
                        delta2 = - np.matmul(jacobinv2,u_array)
                        alpha = 0.3
                        beta = 0.05
                        var_val = x.copy()
                        unew_array = u_array.copy()
        
                        while np.linalg.norm(unew_array) > np.linalg.norm(u_array + alpha*h*np.dot(jacob, delta2)):
                            upd_x = var_val + h*delta2
                            upd_x = abs(upd_x)
                            model3 = LiSModel(upd_x, I) ## Initialize the model
                            ## Change any values if a new value wants to be used apart from the default ones in the model class
                            model3.update_parameters(**upd_params)
                            unew_array = model3.f(h, x_var[:, -1])
                            h *= beta
                            if h < min_h:
                                break
                            #print(h, I*t[i-1]/3600)
                        
                        # Further Update h (step-size)
                        h_new = max(h*(0.25), min_h) ## Saturate at minimum step size
                        
                    elif max_ratio <= 1.0: ## This indicates step size can be increased 
                        h_new = min(h/(0.75), max_h) ## Saturate at maximum step size
                        
                    else:
                        h_new = h
                        
                else:
                    h_new = h
                
                ## Now we solve as usual the new step size will be implemented in the next iteration
                jacob = jacob + regularization_factor * np.eye(len(x))
                jacobinv = np.linalg.inv(jacob)
                # Calculate new values via Newton-Raphson
                delta = - np.matmul(jacobinv,u_array)
                new_val = x + lamda*delta
                new_val = abs(new_val)
                
                model2 = LiSModel(new_val, I) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model2.update_parameters(**upd_params)
                unew_array = model2.f(h, x_var[:, -1])
    
                # Compute the ratio of actual reduction to predicted reduction
                actual_reduction = np.linalg.norm(u_array) - np.linalg.norm(unew_array)
                predicted_reduction = np.linalg.norm(u_array) - np.linalg.norm(u_array + jacob @ delta)
                ratio = abs(actual_reduction / predicted_reduction)
    
                # Update the damping factor based on the ratio
                n_damp = 2
                if ratio > 1e-3:
                    lamda *= (damping_update_factor**n_damp)
                elif ratio < 1e-4:
                    lamda /= (damping_update_factor**n_damp)
                
                # Ensure the damping factor does not go below the minimum value
                lamda = max(lamda, damping_min)
    
                # Calculate error between new_val and old_guess
                err_list = np.zeros((len(x),))
                epsilon = 1e-10
                for j in range(len(err_list)):
                    err_list[j] = abs((new_val[j] - x[j])/(x[j] + epsilon))
                    
                # Check if all absolute difference/err is smaller than a specified tolerance
                tol = 1e-2
                ## Get the V_index (Voltage index from func.py)
                V_index = func.V_index
                
                if np.all(err_list<tol):
                    # If the error is small values have converged hence, update the new variables and break the while loop
                    t_next = t_current + h_new
                    t.append(t_next)
                    h_step.append(h_new)
                    jacob_array.append(norm_jacob)
                    
                    ## Voltage saturation ##
                    if state == 'Discharge':
                        new_val[V_index] = max(new_val[V_index], break_voltage)
                    elif state == 'Charge':
                        new_val[V_index] = min(new_val[V_index], break_voltage)
                        
                    ## Concatenate the new values to the exsisting list
                    x_var = np.concatenate((x_var, new_val[:, np.newaxis]), axis=1)
                    break
    
                else:
                    # Update guess values to be new values
                    ## Voltage saturation ##
                    if state == 'Discharge':
                        new_val[V_index] = max(new_val[V_index], break_voltage)
                    elif state == 'Charge':
                        new_val[V_index] = min(new_val[V_index], break_voltage)
                        
                    x_varguess = new_val 
    
            b = b + 1
            i = i + 1
    
            #print(f"No of iterations: {b}/{int(t_end/h)}")
            
            ### Handle breakpoint for discharge state ###
            ## Get the V_index (Voltage index from func.py)
            V_index = func.V_index
            V_break = x_var[V_index][i-1]
            
            if (V_break<=break_voltage) and (state == 'Discharge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var[:, :breakpoint1]
                t_array = np.asarray(t[:breakpoint1])
            
                return(np.vstack((var_array, t_array)))
                break
            
            ### Handle breakpoint for charge state ###
            elif (V_break>=break_voltage) and (state == 'Charge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var[:, :breakpoint1]
                t_array = np.asarray(t[:breakpoint1])
            
                return(np.vstack((var_array, t_array)))
                break
            
            elif (t[-1] >= t_end):
    # =============================================================================
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the same order as variable list in func.py with additional time array at the last index")
    # =============================================================================
                var_array = x_var
                t_array = np.asarray(t)
            
                return(np.vstack((var_array, t_array)))
                break
        
        ## When a warning is encountered break the loop: ##
        except RuntimeWarning as warning:
            print("Runtime Warning Encountered at Capacity:", t[i-1]/3600)
            warnings.warn(warning)  # Raise the RuntimeWarning again
            break
