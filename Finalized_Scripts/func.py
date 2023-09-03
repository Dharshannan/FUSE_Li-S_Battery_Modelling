from sympy import symbols, diff, exp, sqrt, lambdify

# =============================================================================
# Helper Functions Below:
# =============================================================================
## Derivative function:
def var_func_der(var_list, u, sym):
    # Returns a list of all the derivative
    # List is lambdified w.r.t to the symbols defined (sym)
    der_list = []
    for i in range(len(var_list)):
        der = diff(u, var_list[i])
        der = lambdify(sym, der, 'numpy')
        der_list.append(der)
        
    # Return the lambdified derivative list
    return der_list

## Function to lambdify u array
def u_func_lambdify(u_list, sym):
    for i in range(len(u_list)):
        u_list[i] = lambdify(sym, u_list[i], 'numpy')
        
    return u_list

## Function to return Voltage variable index:
def find_index_with_v(input_list):
    for idx, item in enumerate(input_list):
        if "V" in item:
            return idx
    return -1

# =============================================================================
#               This is the new 2023 3-Stage Model Formulation
# =============================================================================

## Define all parameters as symbols
F = symbols('F') 
Ms = symbols('Ms')
nH = symbols('nH')
nM = symbols('nM')
nL = symbols('nL')
ns8 = symbols('ns8')
R = symbols('R')
ps = symbols('ps') # rho_s
a = symbols('a')
v = symbols('v')
EH0 = symbols('EH0')
EM0 = symbols('EM0')
EL0= symbols('EL0')
jH0 = symbols('jH0')
jM0 = symbols('jM0')
jL0 = symbols('jL0')
CT0 = symbols('CT0')
D8 = symbols('D8')
D4 = symbols('D4')
D2 = symbols('D2')
D1 = symbols('D1')
DLi = symbols('DLi')
kp = symbols('kp')
ks = symbols('ks')
Ksp = symbols('Ksp')
T = symbols('T')

# Define variable symbols

## Cathode Variables
Li_cath = symbols('Li_cath')
s8_cath = symbols('s8_cath')
s4_cath = symbols('s4_cath')
s2_cath = symbols('s2_cath')
s1_cath = symbols('s1_cath')
sp_cath = symbols('sp_cath')
## Seperator Variables
Li_sep = symbols('Li_sep')
s8_sep = symbols('s8_sep')
s4_sep = symbols('s4_sep')
s2_sep = symbols('s2_sep')
s1_sep = symbols('s1_sep')
## Voltage
V = symbols('V')
## Prev_vars
Li_cath_prev = symbols('Li_cath_prev')
s8_cath_prev = symbols('s8_cath_prev')
s4_cath_prev = symbols('s4_cath_prev')
s2_cath_prev = symbols('s2_cath_prev')
s1_cath_prev = symbols('s1_cath_prev')
sp_cath_prev = symbols('sp_cath_prev')
Li_sep_prev = symbols('Li_sep_prev')
s8_sep_prev = symbols('s8_sep_prev')
s4_sep_prev = symbols('s4_sep_prev')
s2_sep_prev = symbols('s2_sep_prev')
s1_sep_prev = symbols('s1_sep_prev')
V_prev = symbols('V_prev')
## h and I
h = symbols('h')
I = symbols('I')

# =============================================================================
# *** NOTE: DO NOT CHANGE THE LIST NAMES AND V_INDEX NAME BELOW AS THEY ARE 
# REFERENCED IN THE SOLVER CLASS AND SOLVER FUNCTION SCRIPT ***
# =============================================================================
## Define lists to pass variable
var_list = [Li_cath, s8_cath, s4_cath, s2_cath, s1_cath, sp_cath, Li_sep, s8_sep, s4_sep, s2_sep, s1_sep, V]
prev_var = [Li_cath_prev, s8_cath_prev, s4_cath_prev, s2_cath_prev, s1_cath_prev, sp_cath_prev,
            Li_sep_prev, s8_sep_prev, s4_sep_prev, s2_sep_prev, s1_sep_prev, V_prev]
## Both these list (var_list and prev_var) must have the same order
h_I = [h, I]
param_list = [F, Ms, nH, nM, nL, ns8, R, ps, a, v, EH0, EM0, EL0, jH0, jM0, jL0,
              CT0, D8, D4, D2, D1, DLi, kp, ks, Ksp, T]
sym = tuple(var_list + prev_var + h_I + param_list)

## Here we check the index of the Voltage variable
V_index = find_index_with_v([str(item) for item in var_list])

# =============================================================================
# ## Now we define the dependant equations before the ODEs
# =============================================================================

jHc = ((Li_cath)**4)*s8_cath*exp(-nH*F*(V-EH0)/(2*R*T))
jHa = ((s4_cath)**2)*exp(nH*F*(V-EH0)/(2*R*T))
jMc = ((Li_cath)**2)*s4_cath*exp(-nM*F*(V-EM0)/(2*R*T))
jMa = ((s2_cath)**2)*exp(nM*F*(V-EM0)/(2*R*T))
jLc = ((Li_cath)**2)*s2_cath*exp(-nL*F*(V-EL0)/(2*R*T))
jLa = ((s1_cath)**2)*exp(nL*F*(V-EL0)/(2*R*T))

iH = a*jH0*(jHc - jHa)
iM = a*jM0*(jMc - jMa)
iL = a*jL0*(jLc - jLa)

CT = Li_cath + s8_cath + s4_cath + s2_cath + s1_cath + Li_sep + s8_sep + s4_sep + s2_sep + s1_sep
D8_dyn = D8*(CT0/CT)
D4_dyn = D4*(CT0/CT)
D2_dyn = D2*(CT0/CT)
D1_dyn = D1*(CT0/CT)
DLi_dyn = DLi*(CT0/CT)

## Now we define the Backward Euler Equations:
# =============================================================================
# u1 is the discretised function for Li_cath (time-dependent) and Jacobian Elements 
# =============================================================================
k_Li_cath = (-I/(v*F)) - DLi_dyn*(Li_cath - Li_sep)

u1 = h*k_Li_cath - Li_cath + Li_cath_prev

u1_ders = var_func_der(var_list, u1, sym)

# =============================================================================
# u2 is the discretised function for s8_cath (time-dependent) and Jacobian Elements
# =============================================================================
k_s8_cath = (-iH/(nH*v*F)) - D8_dyn*(s8_cath - s8_sep)

u2 = h*k_s8_cath - s8_cath + s8_cath_prev

u2_ders = var_func_der(var_list, u2, sym)

# =============================================================================
# u3 is the discretised function for s4_cath (time-dependent) and Jacobian Elements  
# =============================================================================
k_s4_cath = (2*iH/(nH*v*F)) - (iM/(nM*v*F)) - D4_dyn*(s4_cath - s4_sep)

u3 = h*k_s4_cath - s4_cath + s4_cath_prev

u3_ders = var_func_der(var_list, u3, sym)

# =============================================================================
# u4 is the discretised function for s2_cath (time-dependent) and Jacobian Elements  
# =============================================================================
k_s2_cath = (2*iM/(nM*v*F)) - (iL/(nL*v*F)) - D2_dyn*(s2_cath - s2_sep)

u4 = h*k_s2_cath - s2_cath + s2_cath_prev

u4_ders = var_func_der(var_list, u4, sym)

# =============================================================================
# u5 is the discretised function for s1_cath (time-dependent) and Jacobian Elements    
# =============================================================================
k_s1_cath = (2*iL/(nL*v*F)) - D1_dyn*(s1_cath - s1_sep) - (Ms/ps)*kp*sp_cath*(s1_cath - Ksp)

u5 = h*k_s1_cath - s1_cath + s1_cath_prev

u5_ders = var_func_der(var_list, u5, sym)

# =============================================================================
# u6 is the discretised function for sp_cath (time-dependent) and Jacobian Elements   
# =============================================================================
k_sp_cath = (Ms/ps)*kp*sp_cath*(s1_cath - Ksp)

u6 = h*k_sp_cath - sp_cath + sp_cath_prev

u6_ders = var_func_der(var_list, u6, sym)

# =============================================================================
# u7 is the discretised function for Li_sep (time-dependent) and Jacobian Elements
# =============================================================================
k_Li_sep = (I/(v*F)) + DLi_dyn*(Li_cath - Li_sep)

u7 = h*k_Li_sep - Li_sep + Li_sep_prev

u7_ders = var_func_der(var_list, u7, sym)

# =============================================================================
# u8 is the discretised function for s8_sep (time-dependent) and Jacobian Elements
# =============================================================================
k_s8_sep = D8_dyn*(s8_cath - s8_sep) - ks*s8_sep

u8 = h*k_s8_sep - s8_sep + s8_sep_prev

u8_ders = var_func_der(var_list, u8, sym)

# =============================================================================
# u9 is the discretised function for s4_sep (time-dependent) and Jacobian Elements
# =============================================================================
k_s4_sep = D4_dyn*(s4_cath - s4_sep) + 2*ks*s8_sep

u9 = h*k_s4_sep - s4_sep + s4_sep_prev

u9_ders = var_func_der(var_list, u9, sym)

# =============================================================================
# u10 is the discretised function for s2_sep (time-dependent) and Jacobian Elements
# =============================================================================
k_s2_sep = D2_dyn*(s2_cath - s2_sep)

u10 = h*k_s2_sep - s2_sep + s2_sep_prev

u10_ders = var_func_der(var_list, u10, sym)

# =============================================================================
# u11 is the discretised function for s1_sep (time-dependent) and Jacobian Elements
# =============================================================================
k_s1_sep = D1_dyn*(s1_cath - s1_sep)

u11 = h*k_s1_sep - s1_sep + s1_sep_prev

u11_ders = var_func_der(var_list, u11, sym)

# =============================================================================
# u12 is the discretised function for V (non-time-dependent) and Jacobian Elements
# =============================================================================
# This is the Algebraic Constraint
u12 = I - iH - iM - iL

u12_ders = var_func_der(var_list, u12, sym)

# =============================================================================
# ## Update the u_list and jacob_list with the u values and differentials ##
# ## DO NOT CHANGE THE NAMES OF THE LIST i.e. u_list and jacob_list ##
# =============================================================================

u_list = [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12] # Un-lambdified list
u_list = u_func_lambdify(u_list, sym) # Lamdify the u_list
jacob_list = [u1_ders, u2_ders, u3_ders, u4_ders, u5_ders, 
              u6_ders, u7_ders, u8_ders, u9_ders, u10_ders, u11_ders, u12_ders]