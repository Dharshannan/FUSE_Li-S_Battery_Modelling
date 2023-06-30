import numpy as np
import matplotlib.pyplot as plt

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
tt = 7500
h = 5e-3
t = np.arange(0,tt+h,h)
s8 = np.zeros((len(t),))
s4 = np.zeros((len(t),))
s2 = np.zeros((len(t),))
s = np.zeros((len(t),))
sp = np.zeros((len(t),))
V = np.zeros((len(t),))
ratio_arr = np.zeros((len(t),)) # This is to keep track of how the ratio of actual/predicted reduction changes
predicted_arr = np.zeros((len(t),))
actual_arr = np.zeros((len(t),))
jacob_max = np.zeros((len(t),))
jacob_min = np.zeros((len(t),))

## Initialize the variable values
s8[0] = 2.6892000000000003
s4[0] = 0.0027
s2[0] = 0.002697299116926997
s[0] = 8.83072852310722e-10
sp[0] = 2.7e-6
V[0] = 2.430277479547109
ratio_arr[0] = 0.172
predicted_arr[0] = 0
actual_arr[0] = 0
jacob_max[0] = 0
jacob_min[0] = 0
b = 0
breakpoint1 = 0
## Now start time iteration loop
for i in range(1,len(t)):
    # Initialize with guess values for each variable (use previous values as initial guesses)
    s8g = s8[i-1] # s8g stands for sulfur 8 guess value
    s4g = s4[i-1]
    s2g = s2[i-1]
    sg = s[i-1]
    spg = sp[i-1]
    Vg = V[i-1]
    a = 0
    # Define the damping to be 1 at the start, we will dynamically update this if necessary
    lamda = 1 # Damping Factor
    damping_update_factor = 0.5
    damping_min = 1e-8
    regularization_factor = 5e-4
    
    ## Initiate While loop for Newton Raphson
    while True:
    
        # Now calculate u (function) values and define jacobian elements
        
    # =============================================================================
    #   u1 and its jacobian elements:
    # =============================================================================
        u1 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                - ks*s8g) - s8g + s8[i-1]
    
        du1ds8 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*(s8g**3)))*(s4g)\
            - 0.5*(np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh/s8g))))\
                - ks) - 1
        
        du1ds4 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))\
            + (np.exp(2*F*(EH0-Vg)/(R*T))/(s4g**2))*(np.sqrt(fh*s8g)))))
        
        du1dV = h*((((ns8*Ms8*iH0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            + ((2*F)/(R*T))*(np.exp(2*F*(EH0-Vg)/(R*T))/(s4g))*(np.sqrt(fh*s8g)))))
        
    # =============================================================================
    #   u2 and its jacobian elements:
    # =============================================================================
        u2 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
            - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
                + ks*s8g + (((ns4*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                    - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s4g + s4[i-1]
        
        du2ds8 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*(s8g**3)))*(s4g)\
            - 0.5*(np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh/s8g))))
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
    #   u3 and its jacobian elements:
    # =============================================================================
        u3 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))) - s2g + s2[i-1]
        
        du3ds4 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(-0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*(s4g**3)))*(np.sqrt(s2g*(sg**2)))\
            - 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl/s4g)))))
        
        du3ds2 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(0.5*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g*s2g))*(sg)\
            + 0.5*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt((s2g**3)*(sg**2))))*(np.sqrt(fl*s4g))))) - 1
        
        du3ds = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g))\
            + (np.exp(2*F*(EL0-Vg)/(R*T))/((np.sqrt(s2g))*(sg**2)))*(np.sqrt(fl*s4g)))))
        
        du3dV = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*(((2*F)/(R*T))*(np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            + ((2*F)/(R*T))*(np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g)))))
        
    # =============================================================================
    #   u4 and its jacobian elements
    # =============================================================================
        u4 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
            - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))\
                - ((kp*spg)/(v*ps))*(sg - s_sat)) - sg + s[i-1]
        
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
    #   u5 and its jacobian elements
    # =============================================================================
        u5 = I + (iH0*ar*((np.exp(2*F*(Vg-EH0)/(R*T))/np.sqrt(fh*s8g))*(s4g)\
                          - (np.exp(2*F*(EH0-Vg)/(R*T))/s4g)*(np.sqrt(fh*s8g))))\
            + (iL0*ar*((np.exp(2*F*(Vg-EL0)/(R*T))/np.sqrt(fl*s4g))*(np.sqrt(s2g*(sg**2)))\
                       - (np.exp(2*F*(EL0-Vg)/(R*T))/(np.sqrt(s2g*(sg**2))))*(np.sqrt(fl*s4g))))
        
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
    #   u6 and its jacobian elements
    # =============================================================================
        u6 = ((h*kp*spg)/(v*ps))*(sg - s_sat) - spg + sp[i-1]
        
        du6ds = ((h*kp*spg)/(v*ps))
        
        du6dsp = ((h*kp)/(v*ps))*(sg - s_sat) - 1
        
    # =============================================================================
    #   Now we build the jacobian matrix
    # =============================================================================
        jacob = np.array([[du1ds8, du1ds4, 0, 0, du1dV, 0], [du2ds8, du2ds4, du2ds2, du2ds, du2dV, 0], 
                          [0, du3ds4, du3ds2, du3ds, du3dV, 0], [0, du4ds4, du4ds2, du4ds, du4dV, du4dsp],
                          [du5ds8, du5ds4, du5ds2, du5ds, du5dV, 0], [0, 0, 0, du6ds, 0, du6dsp]])
        
        ## Add a regularization factor to the Jacobian to prevent singularity:
        jacob = jacob + regularization_factor * np.eye(jacob.shape[0])
        
        jacobinv = np.linalg.inv(jacob)
        
        # Now define the vector matrices required for Backwards Euler + Newton-Raphson method
        u_array = np.array([u1, u2, u3, u4, u5, u6])
        
        # Create a list to store the old guess values
        old_guess = np.array([s8g, s4g, s2g, sg, Vg, spg])
        
        # Calculate new values via Newton-Raphson
        delta = - np.matmul(jacobinv,u_array)
        new_val = old_guess + lamda*delta
        new_val = abs(new_val)
        
        # New variable values for calculation of new function values
        s8g2 = new_val[0]
        s4g2 = new_val[1]
        s2g2 = new_val[2]
        sg2 = new_val[3]
        Vg2 = new_val[4]
        spg2 = new_val[5]
    
# =============================================================================       
# =============================================================================
        # Now we calculate the new function values for u1,u2,u3,u4,u5,u6
        u1_new = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg2-EH0)/(R*T))/np.sqrt(fh*s8g2))*(s4g2)\
            - (np.exp(2*F*(EH0-Vg2)/(R*T))/s4g2)*(np.sqrt(fh*s8g2))))\
                - ks*s8g2) - s8g2 + s8[i-1]
        
        u2_new = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((np.exp(2*F*(Vg2-EH0)/(R*T))/np.sqrt(fh*s8g2))*(s4g2)\
            - (np.exp(2*F*(EH0-Vg2)/(R*T))/s4g2)*(np.sqrt(fh*s8g2))))\
                + ks*s8g2 + (((ns4*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg2-EL0)/(R*T))/np.sqrt(fl*s4g2))*(np.sqrt(s2g2*(sg2**2)))\
                    - (np.exp(2*F*(EL0-Vg2)/(R*T))/(np.sqrt(s2g2*(sg2**2))))*(np.sqrt(fl*s4g2))))) - s4g2 + s4[i-1]
        
        u3_new = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg2-EL0)/(R*T))/np.sqrt(fl*s4g2))*(np.sqrt(s2g2*(sg2**2)))\
            - (np.exp(2*F*(EL0-Vg2)/(R*T))/(np.sqrt(s2g2*(sg2**2))))*(np.sqrt(fl*s4g2))))) - s2g2 + s2[i-1]
        
        u4_new = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((np.exp(2*F*(Vg2-EL0)/(R*T))/np.sqrt(fl*s4g2))*(np.sqrt(s2g2*(sg2**2)))\
            - (np.exp(2*F*(EL0-Vg2)/(R*T))/(np.sqrt(s2g2*(sg2**2))))*(np.sqrt(fl*s4g2))))\
                - ((kp*spg2)/(v*ps))*(sg2 - s_sat)) - sg2 + s[i-1]
        
        u5_new = I + iH0*ar*((np.exp(2*F*(Vg2-EH0)/(R*T))/np.sqrt(fh*s8g2))*(s4g2)\
            - (np.exp(2*F*(EH0-Vg2)/(R*T))/s4g2)*(np.sqrt(fh*s8g2)))\
        + iL0*ar*((np.exp(2*F*(Vg2-EL0)/(R*T))/np.sqrt(fl*s4g2))*(np.sqrt(s2g2*(sg2**2)))\
            - (np.exp(2*F*(EL0-Vg2)/(R*T))/(np.sqrt(s2g2*(sg2**2))))*(np.sqrt(fl*s4g2)))
        
        u6_new = ((h*kp*spg2)/(v*ps))*(sg2 - s_sat) - spg2 + sp[i-1]
        
        ## Array of new function values
        unew_array = np.array([u1_new, u2_new, u3_new, u4_new, u5_new, u6_new])
        
# =============================================================================       
# =============================================================================

        # Compute the ratio of actual reduction to predicted reduction
        actual_reduction = np.linalg.norm(u_array) - np.linalg.norm(unew_array)
        predicted_reduction = np.linalg.norm(u_array) - np.linalg.norm(u_array + jacob @ delta)
        ratio = abs(actual_reduction / predicted_reduction)
# =============================================================================
#         if abs(predicted_reduction) < 1e-6:
#             ratio = 1  # Set a default value or handle it as desired
#         else:
#             ratio = actual_reduction / predicted_reduction
# =============================================================================
        # Update the damping factor based on the ratio
        n_damp = 1
        if ratio > 1e-3:
            lamda *= damping_update_factor**n_damp
        elif ratio < 1e-4:
            lamda /= damping_update_factor**n_damp
        
        # Ensure the damping factor does not go below the minimum value
        lamda = max(lamda, damping_min)
        
        correction_val = 0 # Correction to avoid divergence
        # For debugging:
        #a = a + 1
        #print(max(u_array))
# =============================================================================
#         if a>500:
#             print(new_val)
#             break
# =============================================================================
        # Calculate error between new_val and old_guess
        err_list = np.zeros((len(old_guess),))
        epsilon = 1e-10
        for j in range(len(err_list)):
            err_list[j] = abs((new_val[j] - (old_guess[j] - correction_val))/(old_guess[j] + epsilon))
            
        # Check if all absolute difference/err is smaller than a specified tolerance
        tol = 1e-2
        if np.all(err_list<tol):
            # If the error is small values have converged hence, update the new variables and break the while loop
            s8[i] = new_val[0] 
            s4[i] = new_val[1] 
            s2[i] = new_val[2] 
            s[i] = new_val[3] 
            V[i] = min(new_val[4],2.43027748) 
            sp[i] = new_val[5] 
            ratio_arr[i] = ratio
            predicted_arr[i] = predicted_reduction
            actual_arr[i] = actual_reduction
            jacob_max[i] = np.max(jacob)
            jacob_min[i] = np.min(jacob)
            print("ratio:",ratio)
            print("Determinant:",np.linalg.det(jacob))
            print("Max func value:",max(unew_array))
            break
        else:
            # Update guess values to be new values
            s8g = new_val[0] + correction_val
            s4g = new_val[1] + correction_val
            s2g = new_val[2] + correction_val
            sg = new_val[3] + correction_val
            Vg = min(new_val[4],2.43027748)  + correction_val
            spg = new_val[5] + correction_val
# =============================================================================
#         print("\n")
#         print(new_val)
#         print(old_guess)
# =============================================================================
        #print(u5, du5ds8, du5ds4, du5ds2, du5ds, du5dV)
        #break
    #break
    b = b + 1
    print(f"No of iterations: {b}/{int(tt/h)}")
    if (V[i]<2.241):
        breakpoint1 = i
        break

## Calculate partial current iH and iL values for plotting
EH = EH0 + ((R*T)/(4*F))*np.log(fh*s8/(s4**2))
EL = EL0 + ((R*T)/(4*F))*np.log(fl*s4/(s2*(s**2)))

nH = V - EH
nL = V - EL

iH = -2*iH0*ar*np.sinh((ne*F*nH)/(2*R*T))
iL = -2*iL0*ar*np.sinh((ne*F*nL)/(2*R*T))

# Now we plot:
Ah = (t/3600)*I
#Ah = (3 * ne * F * s8 / (ns8 * Ms8) + ne * F * s4 / (ns4 * Ms8)) / 3600
plt.plot(Ah[:breakpoint1],V[:breakpoint1])
plt.title("Cell Output Voltage vs Discharge Capacity")
plt.ylabel("Potential (V)")
plt.xlabel("Discharge Capacity (Ah)")
plt.xlim([-0.25,3.5])
plt.ylim([2.2,2.45])
plt.show()
# =============================================================================
# plt.plot(Ah,sp, c= 'orange')
# plt.title("Precipitated Sulfur Mass vs Discharge Capacity")
# plt.ylabel("Precipitated S (g)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# #print(max(sp))
# =============================================================================
# =============================================================================
# plt.plot(Ah,iH)
# plt.ylabel("iH (A)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# =============================================================================
# =============================================================================
# plt.plot(Ah,predicted_arr)
# plt.ylabel("Predicted Reduction")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# 
# plt.plot(Ah, actual_arr)
# plt.ylabel("Actual Reduction")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# 
# plt.plot(Ah, ratio_arr, c="orange")
# plt.ylabel("Ratio of Actual/Predicted Reduction")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# 
# plt.plot(Ah, jacob_max, c="orange")
# plt.ylabel("Max Jacobian Values")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.show()
# 
# plt.plot(Ah, jacob_min, c="orange")
# plt.ylabel("Min Jacobian Values")
# plt.xlabel("Discharge Capacity (Ah)")
# =============================================================================
#print(iH + iL)
#print(V)
#print(sp)
