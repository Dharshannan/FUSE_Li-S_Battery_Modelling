import numpy as np
import timeit
import warnings

class LiSModel:
    
    def __init__(self, x, I):
        # Define constants
        self.F = 96485 
        self.Ms8 = 32
        self.ne = 4
        self.ns8 = 8
        self.ns4 = 4
        self.ns2 = 2
        self.ns = 1
        self.R = 8.3145
        self.ps = 2e3
        self.ar = 0.96
        self.fh = 0.73
        self.fl = 0.067
        self.v = 0.0114
        self.EH0 = 2.4
        self.EL0 = 2.0
        self.iH0 = 10
        self.iL0 = 5
        self.s_sat = 1e-4
        self.kp = 100
        self.ks = 2e-4
        self.T = 298
        
        # Store variables
        self.x = x
        self.s8 = x[0]
        self.s4 = x[1]
        self.s2 = x[2]
        self.s = x[3]
        self.V = x[4]
        self.sp = x[5]
        self.I = I
    
    def get_func_vals(self):
        ## Import dependent module
        import func
        
        ## Now we define the RK2 Equations:
        # =============================================================================
        # u1 is the discretised function for s8 (time-dependent) and Jacobian Elements 
        # =============================================================================
        
        self.u1 = func.u1
        
        self.du1ds8 = func.du1ds8
        
        self.du1ds4 = func.du1ds4
        
        self.du1dV = func.du1dV
        
        # =============================================================================
        # u2 is the discretised function for s4 (time-dependent) and Jacobian Elements
        # =============================================================================
        
        self.u2 = func.u2
        
        self.du2ds8 = func.du2ds8
        
        self.du2ds4 = func.du2ds4
        
        self.du2ds2 = func.du2ds2
        
        self.du2ds = func.du2ds
        
        self.du2dV = func.du2dV
        
        # =============================================================================
        # u3 is the discretised function for s2 (time-dependent) and Jacobian Elements  
        # =============================================================================
        
        self.u3 = func.u3
        
        self.du3ds4 = func.du3ds4
        
        self.du3ds2 = func.du3ds2
        
        self.du3ds = func.du3ds
        
        self.du3dV = func.du3dV
        
        # =============================================================================
        # u4 is the discretised function for s (time-dependent) and Jacobian Elements    
        # =============================================================================
        
        self.u4 = func.u4
        
        self.du4ds4 = func.du4ds4
        
        self.du4ds2 = func.du4ds2
        
        self.du4ds = func.du4ds
        
        self.du4dsp = func.du4dsp
        
        self.du4dV = func.du4dV
        
        # =============================================================================
        # u5 is the discretised function for I = iH + iL (Non-time-dependent) and Jacobian Elements   
        # =============================================================================
        ## This does not require k1 and k2 as it is not a time dependent equation
        self.u5 = func.u5
        
        self.du5ds8 = func.du5ds8
        
        self.du5ds4 = func.du5ds4
        
        self.du5ds2 = func.du5ds2
        
        self.du5ds = func.du5ds 
        
        self.du5dV = func.du5dV 
        
        # =============================================================================
        # u6 is the discretised function for sp (time-dependent) (sp is Precipitated Sulfur) and Jacobian Elements
        # =============================================================================
        
        self.u6 = func.u6
        
        self.du6ds = func.du6ds
        
        self.du6dsp = func.du6dsp
    
    ## Now we define f function to get the u1-u6 function values and return in an array
    def f(self, h, s8_pre, s4_pre, s2_pre, s_pre, V_pre, sp_pre):
        ## Call get_func_vals() to get the values from the func.py script ##
        self.get_func_vals()
        x = self.x
        h_arr = [h]
        prev_var = [s8_pre, s4_pre, s2_pre, s_pre, V_pre, sp_pre]
        ## Ensure the params list follows the same order as the list2 in func.py ##
        params = [self.F, self.Ms8, self.ne, self.ns8, self.ns4, self.ns2, self.ns, 
                  self.R, self.ps, self.ar, self.fh, self.fl, 
                  self.v, self.EH0, self.EL0, self.iH0, self.iL0, self.s_sat, self.kp, self.ks, self.T]
        
        x_new = list(x) + h_arr + prev_var + [self.I] + params
        
        ## Now carry out substitution with values of variables
        u1_res = float(self.u1(*x_new))
        u2_res = float(self.u2(*x_new))
        u3_res = float(self.u3(*x_new))
        u4_res = float(self.u4(*x_new))
        u5_res = float(self.u5(*x_new))
        u6_res = float(self.u6(*x_new))   
        
        u_array = np.array([u1_res, u2_res, u3_res, u4_res, u5_res, u6_res], dtype='float64')
        return(u_array)
    
    ## Now define function to return Jacobian Matrix
    def jacobian(self, h):
        ## Call get_func_vals() to get the values from the func.py script ##
        self.get_func_vals()
        x = self.x
        h_arr = [h]
        ## Ensure the params list follows the same order as the list2 in func.py ##
        params = [self.F, self.Ms8, self.ne, self.ns8, self.ns4, self.ns2, self.ns, 
                  self.R, self.ps, self.ar, self.fh, self.fl, 
                  self.v, self.EH0, self.EL0, self.iH0, self.iL0, self.s_sat, self.kp, self.ks, self.T]
        
        x_new = list(x) + h_arr + list(x) + [self.I] + params
        
        # =============================================================================
        # u1 (ds8/dt) Jacobian Elements 
        # =============================================================================
        du1ds8_res = float(self.du1ds8(*x_new))
        du1ds4_res = float(self.du1ds4(*x_new))
        du1dV_res = float(self.du1dV(*x_new))
        
        # =============================================================================
        # u2 (ds4/dt) Jacobian Elements 
        # =============================================================================
        du2ds8_res = float(self.du2ds8(*x_new))
        du2ds4_res = float(self.du2ds4(*x_new))
        du2ds2_res = float(self.du2ds2(*x_new))
        du2ds_res = float(self.du2ds(*x_new))
        du2dV_res = float(self.du2dV(*x_new))
        
        # =============================================================================
        # u3 (ds2/dt) Jacobian Elements 
        # =============================================================================
        du3ds4_res = float(self.du3ds4(*x_new))
        du3ds2_res = float(self.du3ds2(*x_new))
        du3ds_res = float(self.du3ds(*x_new))
        du3dV_res = float(self.du3dV(*x_new))
        
        # =============================================================================
        # u4 (ds/dt) Jacobian Elements 
        # =============================================================================
        du4ds4_res = float(self.du4ds4(*x_new))
        du4ds2_res = float(self.du4ds2(*x_new))
        du4ds_res = float(self.du4ds(*x_new))
        du4dsp_res = float(self.du4dsp(*x_new))
        du4dV_res = float(self.du4dV(*x_new))
        
        # =============================================================================
        # u5 (V) Jacobian Elements 
        # =============================================================================
        du5ds8_res = float(self.du5ds8(*x_new))
        du5ds4_res = float(self.du5ds4(*x_new))
        du5ds2_res = float(self.du5ds2(*x_new))
        du5ds_res = float(self.du5ds(*x_new))
        du5dV_res = float(self.du5dV(*x_new))
        
        # =============================================================================
        # u6 (dsp/dt) Jacobian Elements 
        # =============================================================================
        du6ds_res = float(self.du6ds(*x_new))
        du6dsp_res = float(self.du6dsp(*x_new))

        # =============================================================================
        # Now formulate Jacobian array             
        # =============================================================================
        jacob = np.array([[du1ds8_res, du1ds4_res, 0, 0, du1dV_res, 0], [du2ds8_res, du2ds4_res, du2ds2_res, du2ds_res, du2dV_res, 0],\
                          [0, du3ds4_res, du3ds2_res, du3ds_res, du3dV_res, 0], [0, du4ds4_res, du4ds2_res, du4ds_res, du4dV_res, du4dsp_res],\
                          [du5ds8_res, du5ds4_res, du5ds2_res, du5ds_res, du5dV_res, 0], [0, 0, 0, du6ds_res, 0, du6dsp_res]], dtype='float64')
        
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

def LiS_Solver(s8i, s4i, s2i, si, Vi, spi,  
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
    s8 = [s8i]
    s4 = [s4i]
    s2 = [s2i]
    s = [si]
    V = [Vi]
    sp = [spi]
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
    start = timeit.default_timer() ## Start timer
    while t[-1] < t_end:
        
        try:
            # Initialize with guess values for each variable (use previous values as initial guesses)
            s8guess = s8[i-1] # s8g stands for sulfur 8 guess value
            s4guess = s4[i-1]
            s2guess = s2[i-1]
            sguess = s[i-1]
            spguess = sp[i-1]
            Vguess = V[i-1]
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
                x = np.array([s8guess, s4guess, s2guess, sguess, Vguess, spguess])
                
                ## Call model class and get the function arrays and jacobian
                model = LiSModel(x, I) ## Initialize the model
                ## Change any values if a new value wants to be used apart from the default ones in the model class
                model.update_parameters(**upd_params)
                ## Change parameter values if recursion has happenend (Backtrack to set value)
                if backtracked == True:
                    model.update_parameters(**params_backtrack)
                ## Now solve as usual with new backtracked parameter value
                u_array = model.f(h, s8[-1], s4[-1], s2[-1], s[-1], V[-1], sp[-1])
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
                        new_guess = LiS_Solver(s8guess, s4guess, s2guess, sguess, Vguess, spguess, 
                                               t_end2, h, I, break_voltage, state=state, t0=t02, backtracked=True, 
                                               params_backtrack=params_backtrack, upd_params=upd_params)
                        
                        ## Now update the guess values and run solver by updating u_array and jacobian
                        x_upd = np.array([new_guess[0][-1], new_guess[1][-1], new_guess[2][-1], 
                                          new_guess[3][-1], new_guess[4][-1], new_guess[5][-1]])
                        # Update the model
                        model = LiSModel(x_upd, I)
                        ## Change any values if a new value wants to be used apart from the default ones in the model class
                        model.update_parameters(**upd_params)
                        u_array = model.f(h, s8[-1], s4[-1], s2[-1], s[-1], V[-1], sp[-1])
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
                            unew_array = model3.f(h, s8[-1], s4[-1], s2[-1], s[-1], V[-1], sp[-1])
                            h *= beta
                            if h< min_h:
                                break
                            #print(h, I*t[i-1]/3600)
                        
                        # Further Update h (step-size)
                        h_new = max(h*(0.6), min_h) ## Saturate at minimum step size
                        
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
                unew_array = model2.f(h, s8[-1], s4[-1], s2[-1], s[-1], V[-1], sp[-1])
    
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
                
                correction_val = 0 # Correction to avoid divergence
    
                # Calculate error between new_val and old_guess
                err_list = np.zeros((len(x),))
                epsilon = 1e-10
                for j in range(len(err_list)):
                    err_list[j] = abs((new_val[j] - (x[j] - correction_val))/(x[j] + epsilon))
                    
                # Check if all absolute difference/err is smaller than a specified tolerance
                tol = 1e-2
                if np.all(err_list<tol):
                    # If the error is small values have converged hence, update the new variables and break the while loop
                    t_next = t_current + h_new
                    t.append(t_next)
                    h_step.append(h_new)
                    jacob_array.append(norm_jacob)
                    s8.append(new_val[0]) 
                    s4.append(new_val[1])
                    s2.append(new_val[2])
                    s.append(new_val[3])
                    ## NOTE: These are handled differently in the solver script for microcycling
                    if state == 'Discharge':
                        V.append(min(new_val[4],V[0]))
                    elif state == 'Charge':
                        V.append(max(new_val[4],V[0]))
                    sp.append(new_val[5])
                    #print(new_val)
                    break
    
                else:
                    # Update guess values to be new values
                    s8guess = new_val[0] + correction_val
                    s4guess = new_val[1] + correction_val
                    s2guess = new_val[2] + correction_val
                    sguess = new_val[3] + correction_val
                    ## NOTE: These are handled differently in the solver script for microcycling
                    if state == 'Discharge':
                        Vguess = min(new_val[4],V[0])  + correction_val
                    elif state == 'Charge':
                        Vguess = max(new_val[4],V[0])  + correction_val
                    spguess = new_val[5] + correction_val
    
            b = b + 1
            i = i + 1
    
            #print(f"No of iterations: {b}/{int(t_end/h)}")
            ### Change breakpoint voltage accordingly ###
            
            # Handle breakpoint for discharge state
            if (V[i-1]<break_voltage) and (state == 'Discharge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the form: [s8_array, s4_array, s2_array, s_array, V_array, sp_array, time_array]")
    # =============================================================================
                var_array = np.array([s8[:breakpoint1], s4[:breakpoint1], s2[:breakpoint1],
                                     s[:breakpoint1], V[:breakpoint1], sp[:breakpoint1], t[:breakpoint1]])
            
                return(var_array)
                break
            
            # Handle breakpoint for charge state
            elif (V[i-1]>break_voltage) and (state == 'Charge'):
                breakpoint1 = i-1
    # =============================================================================
    #             print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the form: [s8_array, s4_array, s2_array, s_array, V_array, sp_array, time_array]")
    # =============================================================================
                var_array = np.array([s8[:breakpoint1], s4[:breakpoint1], s2[:breakpoint1],
                                     s[:breakpoint1], V[:breakpoint1], sp[:breakpoint1], t[:breakpoint1]])
            
                return(var_array)
                break
            
            elif (t[-1] >= t_end):
    # =============================================================================
    #             print("The time taken for completion :", timeit.default_timer() - start, "s")
    #             print("Solved array returned in the form: [s8_array, s4_array, s2_array, s_array, V_array, sp_array, time_array]")
    # =============================================================================
                var_array = np.array([s8, s4, s2, s, V, sp, t])
            
                return(var_array)
                break
        
        ## When a warning is encountered break the loop: ##
        except RuntimeWarning as warning:
            print("Runtime Warning Encountered at Capacity:", I*t[i-1]/3600)
            warnings.warn(warning)  # Raise the RuntimeWarning again
            break