import numpy as np
import functools
import timeit

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
        self.EH0 = 2.35
        self.EL0 = 2.195
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
        
    @functools.lru_cache(maxsize=None)
    def f(self, h, s8_arr, s4_arr, s2_arr, s_arr, V_arr, sp_arr):
        # _arr stands for the previous variable value (Backward Euler Discretisation):
        # Ex: -s8g + s8_arr is basically the discretisation: -s8[j] + s8[j-1]
        # Get constant values from initialised model
        F = self.F
        Ms8 = self.Ms8
        ne = self.ne
        ns8 = self.ns8
        ns4 = self.ns4
        ns2 = self.ns2
        ns = self.ns
        R = self.R
        ps = self.ps
        ar = self.ar
        fh = self.fh
        fl = self.fl
        v = self.v
        EH0 = self.EH0
        EL0 = self.EL0
        iH0 = self.iH0
        iL0 = self.iL0
        s_sat = self.s_sat
        kp = self.kp
        ks = self.ks
        T = self.T
        
        # Variable values
        s8g = self.s8
        s4g = self.s4
        s2g = self.s2
        sg = self.s
        Vg = self.V
        spg = self.sp
        I = self.I
        
        ## Define the u functions 
        # =============================================================================
        # u1 is the discretised function for s8 (time-dependent)   
        # =============================================================================
        u1 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                - ks*s8g) - s8g + s8_arr
        
        # =============================================================================
        # u2 is the discretised function for s4 (time-dependent)   
        # =============================================================================
        u2 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                + ks*s8g + (((ns4*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                    - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s4g + s4_arr
        
        # =============================================================================
        # u3 is the discretised function for s2 (time-dependent)   
        # =============================================================================
        u3 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s2g + s2_arr
        
        # =============================================================================
        # u4 is the discretised function for s (time-dependent)   
        # =============================================================================
        u4 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))\
                - ((kp*spg)/(v*ps))*(sg - s_sat)) - sg + s_arr
        
        # =============================================================================
        # u5 is the discretised function for I = iH + iL (Non-time-dependent)   
        # =============================================================================
        u5 = I + iH0*ar*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g)))\
        + iL0*ar*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))
        
        # =============================================================================
        # u6 is the discretised function for sp (time-dependent) (sp is Precipitated Sulfur)
        # =============================================================================
        u6 = ((h*kp*spg)/(v*ps))*(sg - s_sat) - spg + sp_arr
        
        u_array = np.array([u1, u2, u3, u4, u5, u6])
        
        return(u_array)
    
    @functools.lru_cache(maxsize=None)
    def jacobian(self, h):
        # Get constant values from initialised model
        F = self.F
        Ms8 = self.Ms8
        ne = self.ne
        ns8 = self.ns8
        ns4 = self.ns4
        ns2 = self.ns2
        ns = self.ns
        R = self.R
        ps = self.ps
        ar = self.ar
        fh = self.fh
        fl = self.fl
        v = self.v
        EH0 = self.EH0
        EL0 = self.EL0
        iH0 = self.iH0
        iL0 = self.iL0
        s_sat = self.s_sat
        kp = self.kp
        ks = self.ks
        T = self.T
        
        # Variable values
        s8g = self.s8
        s4g = self.s4
        s2g = self.s2
        sg = self.s
        Vg = self.V
        spg = self.sp
        
        # =============================================================================
        # u1 jacobian elements            
        # =============================================================================
        du1ds8 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*(s8g**3)))*(s4g)\
            - 0.5*(np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh/s8g))))\
                - ks) - 1
        
        du1ds4 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))\
            + (np.exp(2*F*(EH0-Vg)/(R*T))/(s4g**2))*(np.sqrt(fh*s8g)))))
        
        du1dV = h*((((ns8*Ms8*iH0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            + ((2*F)/(R*T))*(np.exp(2*F*(EH0-Vg)/(R*T))/(s4g))*(np.sqrt(fh*s8g)))))
        
        # =============================================================================
        # u2 jacobian elements            
        # =============================================================================
        du2ds8 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*(s8g**3)))*(s4g)\
            - 0.5*(np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh/s8g))))\
                + ks)
        
        du2ds4 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))\
            + (np.exp(2*F*(EH0-Vg)/(R*T))/(s4g**2))*(np.sqrt(fh*s8g)))) + (((ns4*Ms8*iL0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*(s4g**3)))*(np.sqrt(s2g*(sg**2)))\
                - 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl/s4g))))) - 1
                                                                                                      
        du2ds2 = h*((((ns4*Ms8*iL0*ar)/(ne*F))*(0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g*s2g))*(sg)\
            + 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt((s2g**3)*(sg**2))))*(np.sqrt(fl*s4g)))))

        du2ds = h*((((ns4*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g))\
            + (np.exp(2*F*(EL0-Vg)/(R*T))/((np.sqrt(s2g))*(sg**2)))*(np.sqrt(fl*s4g)))))                                                                                  
        
        du2dV = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            + ((2*F)/(R*T))*(np.exp(2*F*(EH0-Vg)/(R*T))/(s4g))*(np.sqrt(fh*s8g)))) + (((ns4*Ms8*iL0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                + ((2*F)/(R*T))*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))))
        
        # =============================================================================
        # u3 jacobian elements            
        # =============================================================================                                                                                                       
        du3ds4 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*(s4g**3)))*(np.sqrt(s2g*(sg**2)))\
            - 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl/s4g)))))
        
        du3ds2 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g*s2g))*(sg)\
            + 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt((s2g**3)*(sg**2))))*(np.sqrt(fl*s4g))))) - 1
        
        du3ds = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g))\
            + (np.exp(2*F*(EL0-Vg)/(R*T))/((np.sqrt(s2g))*(sg**2)))*(np.sqrt(fl*s4g)))))
        
        du3dV = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            + ((2*F)/(R*T))*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))))           
        
        # =============================================================================
        # u4 jacobian elements            
        # ============================================================================= 
        du4ds4 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*(s4g**3)))*(np.sqrt(s2g*(sg**2)))\
            - 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl/s4g)))))
        
        du4ds2 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*(0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g*s2g))*(sg)\
            + 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt((s2g**3)*(sg**2))))*(np.sqrt(fl*s4g)))))
        
        du4ds = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g))\
            + (np.exp(2*F*(EL0-Vg)/(R*T))/((np.sqrt(s2g))*(sg**2)))*(np.sqrt(fl*s4g))))\
                   - ((kp*spg)/(v*ps))) - 1
        
        du4dsp = h*(-kp/(v*ps))*(sg - s_sat)
        
        du4dV = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            + ((2*F)/(R*T))*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))))
        
        # =============================================================================
        # u5 jacobian elements            
        # =============================================================================
        du5ds8 = iH0*ar*(-0.5*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*(s8g**3)))*(s4g)\
            - 0.5*(np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh/s8g)))
        
        du5ds4 = iH0*ar*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))\
            + (np.exp(2*F*(EH0-Vg)/(R*T))/(s4g**2))*(np.sqrt(fh*s8g)))\
        + iL0*ar*(-0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*(s4g**3)))*(np.sqrt(s2g*(sg**2)))\
            - 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl/s4g)))
        
        du5ds2 = iL0*ar*(0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g*s2g))*(sg)\
            + 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt((s2g**3)*(sg**2))))*(np.sqrt(fl*s4g)))
        
        du5ds = iL0*ar*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g))\
            + (np.exp(2*F*(EL0-Vg)/(R*T))/((np.sqrt(s2g))*(sg**2)))*(np.sqrt(fl*s4g)))
        
        du5dV = iH0*ar*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            + ((2*F)/(R*T))*(np.exp(2*F*(EH0-Vg)/(R*T))/(s4g))*(np.sqrt(fh*s8g)))\
        + iL0*ar*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            + ((2*F)/(R*T))*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))
        
        # =============================================================================
        # u6 jacobian elements            
        # =============================================================================
        du6ds = ((h*kp*spg)/(v*ps))
        
        du6dsp = ((h*kp)/(v*ps))*(sg - s_sat) - 1
        # =============================================================================
        # Now formulate Jacobian array             
        # =============================================================================
        jacob = np.array([[du1ds8, du1ds4, 0, 0, du1dV, 0], [du2ds8, du2ds4, du2ds2, du2ds, du2dV, 0],
                          [0, du3ds4, du3ds2, du3ds, du3dV, 0], [0, du4ds4, du4ds2, du4ds, du4dV, du4dsp],
                          [du5ds8, du5ds4, du5ds2, du5ds, du5dV, 0], [0, 0, 0, du6ds, 0, du6dsp]])
        
        return(jacob)
    
# =============================================================================  
# =============================================================================
# =============================================================================
#                Now we define the solver for the model class  
# =============================================================================
# =============================================================================
# =============================================================================

def LiS_Solver(s8i, s4i, s2i, si, Vi, spi,  
               t_end, h0, I, break_voltage, state = None, t0 = 0):
    
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
    min_h = 1e-4 # Minimum step size
    max_h = 1 # Maximum step size
    max_jacobian = 0 # Variable to store maximum jacobian determinant
    
    ## Now start time iteration loop
    i = 1
    start = timeit.default_timer() ## Start timer
    while t[-1] < t_end:
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
        lamda = 1 # Damping Factor
        damping_update_factor = 0.5
        damping_min = 1e-8
        regularization_factor = 5e-4
        
        while True:
        
            # Now calculate u (function) values and define jacobian elements using model class
            
            # Create a list to store the old guess values
            x = np.array([s8guess, s4guess, s2guess, sguess, Vguess, spguess])
            
            ## Call model class and get the function arrays and jacobian
            model = LiSModel(x, I) ## Initialize the model
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
                if max_ratio >= 1.5: ## This indicates step needs to be reduced
                    h_new = max(h*(0.25), min_h) ## Saturate at minimum step size
                elif max_ratio <= 0.2: ## This indicates step size can be increased 
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
                if state == 'Discharge':
                    V.append(min(new_val[4],V[0]))
                elif state == 'Charge':
                    V.append(max(new_val[4],V[0]))
                sp.append(new_val[5])
                break
            else:
                # Update guess values to be new values
                s8guess = new_val[0] + correction_val
                s4guess = new_val[1] + correction_val
                s2guess = new_val[2] + correction_val
                sguess = new_val[3] + correction_val
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