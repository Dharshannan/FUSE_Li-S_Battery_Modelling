import numpy as np
import matplotlib.pyplot as plt
import timeit

## Define the constants and parameters
F = 96485 
Ms8 = 32
ne = 4
ns8 = 8
ns4 = 4
ns2 = 2
ns = 1
R = 8.3145
ps = 2e3
ar = 0.96
fh = 0.73
fl = 0.067
v = 0.0114
EH0 = 2.35
EL0= 2.195
iH0 = 10
iL0 = 5
s_sat = 1e-4
kp = 100
ks = 2e-4
I = 1.7
T = 298

## Define variable arrays
t0 = 0
t_end = 7500
h0 = 0.05

## Initialize the variable values and arrays
s8 = [2.6892000000000003]
s4 = [0.0027]
s2 = [0.002697299116926997]
s = [8.83072852310722e-10]
sp = [2.7e-06]
V = [2.430277479547109]
t = [t0]
h_step = [h0]
jacob_array = []

b = 0
breakpoint1 = 0

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
    
    ## Initiate While loop for Newton Raphson
    while True:
    
        # Now calculate u (function) values and define jacobian elements
        
        # Create a list to store the old guess values
        x = np.array([s8guess, s4guess, s2guess, sguess, Vguess, spguess])
        
        # Now we define all the functions for u1-u6 and jacobian into functions that can be called
        # =============================================================================
        # Function f(x) which computes all the u1-u6 values            
        # =============================================================================
        def f(x):
            s8g = x[0]
            s4g = x[1]
            s2g = x[2]
            sg = x[3]
            Vg = x[4]
            spg = x[5]
            
            u1 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
                - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                    - ks*s8g) - s8g + s8[i-1]
            
            u2 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
                - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                    + ks*s8g + (((ns4*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                        - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s4g + s4[i-1]
            
            u3 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s2g + s2[i-1]
            
            u4 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))\
                    - ((kp*spg)/(v*ps))*(sg - s_sat)) - sg + s[i-1]
            
            u5 = I + iH0*ar*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
                - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g)))\
            + iL0*ar*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))
            
            u6 = ((h*kp*spg)/(v*ps))*(sg - s_sat) - spg + sp[i-1]
            
            u_array = np.array([u1, u2, u3, u4, u5, u6])
            
            return(u_array)
        
        # =============================================================================
        # Function jacobian(x) which computes all the jacobian elements of u1-u6            
        # =============================================================================
        def jacobian(x):
            s8g = x[0]
            s4g = x[1]
            s2g = x[2]
            sg = x[3]
            Vg = x[4]
            spg = x[5]
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
        
        # Compute the u_array using the f(x) function and x (variable array)
        u_array = f(x)
        # Compute Jacobian
        jacob = jacobian(x)
        
        ### Now we calculate the determinant of the Jacobian and check to alter step size ###
        norm_jacob = np.linalg.det(jacob)
        jacob_ratio = 0
        # =============================================================================
        #     ### This will dynamically update the step size every iteration ###
        # =============================================================================
        if i > 1: ## Only check after 1st iteration ##
            jacob_ratio = abs(jacob_array[i-2]/jacob_array[i-3])
            ## These values need configuration
            if jacob_ratio >= 1.2: ## This indicates step needs to be reduced
                h_new = h*(0.25)
            elif jacob_ratio <= 0.75: ## This indicates step size can be increased 
                h_new = h/(0.75)
            elif np.round(jacob_ratio, 3) == 1.0: ## When values do not change too much just set h = 1
                h_new = 1
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
        unew_array = f(new_val)

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
            V.append(min(new_val[4],2.43027748))
            sp.append(new_val[5])
# =============================================================================
#             print("new step size:", h_new)
#             print("Jacobian ratio:", jacob_ratio)
# =============================================================================
# =============================================================================
#             print("ratio:",ratio)
#             print("Determinant:",np.linalg.det(jacob))
#             print("Max func value:",max(unew_array))
# =============================================================================
            break
        else:
            # Update guess values to be new values
            s8guess = new_val[0] + correction_val
            s4guess = new_val[1] + correction_val
            s2guess = new_val[2] + correction_val
            sguess = new_val[3] + correction_val
            Vguess = min(new_val[4],2.43027748)  + correction_val
            spguess = new_val[5] + correction_val

    b = b + 1
    i = i + 1
    print(f"No of iterations: {b}/{int(t_end/h)}")
    ### Change breakpoint voltage accordingly ###
    if (V[i-1]<2.241):
        breakpoint1 = i-1
        print("!!! RUN TERMINATED DUE TO EXPLOSION OF JACOBIAN DETERMINANT !!!")
        break

print("The time taken for completion :", timeit.default_timer() - start, "s") ## Calculate time required to run solution

## Turn all arrays to numpy arrays
s8 = np.array(s8)
s4 = np.array(s4)
s2 = np.array(s2)
s = np.array(s)
V = np.array(V)
sp = np.array(sp)
## Calculate partial current iH and iL values for plotting
EH = EH0 + ((R*T)/(4*F))*np.log(fh*s8[:breakpoint1]/(s4[:breakpoint1]**2))
EL = EL0 + ((R*T)/(4*F))*np.log(fl*s4[:breakpoint1]/(s2[:breakpoint1]*(s[:breakpoint1]**2)))

nH = V[:breakpoint1] - EH
nL = V[:breakpoint1] - EL

iH = -2*iH0*ar*np.sinh((ne*F*nH)/(2*R*T))
iL = -2*iL0*ar*np.sinh((ne*F*nL)/(2*R*T))

# Now we plot:
t = np.array(t)
Ah = (t/3600)*I
#Ah = (3 * ne * F * s8 / (ns8 * Ms8) + ne * F * s4 / (ns4 * Ms8)) / 3600
plt.plot(Ah[:breakpoint1],V[:breakpoint1])
plt.title(f"Cell Output Voltage vs Discharge Capacity {I}A")
plt.ylabel("Potential (V)")
plt.xlabel("Discharge Capacity (Ah)")
plt.xlim([-0.25,3.5])
plt.ylim([2.2,2.45])
#plt.savefig('1.7A_Voltage_Output.png', dpi=1200)
plt.show()

# =============================================================================
# plt.plot(Ah,sp, c= 'orange')
# plt.title("Precipitated Sulfur Mass vs Discharge Capacity")
# plt.ylabel("Precipitated S (g)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# =============================================================================
#print(max(sp))
# =============================================================================
# plt.plot(Ah[:breakpoint1],iH)
# plt.ylabel("iH (A)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# =============================================================================

#print(iH + iL)
#print(V)
#print(sp)
