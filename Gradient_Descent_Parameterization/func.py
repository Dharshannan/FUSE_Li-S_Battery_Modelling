from sympy import symbols, diff, exp, sqrt, lambdify

## Define all parameters as symbols
F = symbols('F') 
Ms8 = symbols('Ms8')
ne = symbols('ne')
ns8 = symbols('ns8')
ns4 = symbols('ns4')
ns2 = symbols('ns2')
ns = symbols('ns')
R = symbols('R')
ps = symbols('ps')
ar = symbols('ar')
fh = symbols('fh')
fl = symbols('fl')
v = symbols('v')
EH0 = symbols('EH0')
EL0= symbols('EL0')
iH0 = symbols('iH0')
iL0 = symbols('iL0')
s_sat = symbols('s_sat')
kp = symbols('kp')
ks = symbols('ks')
T = symbols('T')

# Define variable symbols
s8g = symbols('s8g')
s4g = symbols('s4g')
s2g = symbols('s2g')
sg = symbols('sg')
Vg = symbols('Vg')
spg = symbols('spg')
h = symbols('h')
s8_arr = symbols('s8_arr')
s4_arr = symbols('s4_arr')
s2_arr = symbols('s2_arr')
s_arr = symbols('s_arr')
V_arr = symbols('V_arr')
sp_arr = symbols('sp_arr')
I = symbols('I')

list1 = [s8g, s4g, s2g, sg, Vg, spg, h, s8_arr, s4_arr, s2_arr, s_arr, V_arr, sp_arr, I]
list2 = [F, Ms8, ne, ns8, ns4, ns2, ns, R, ps, ar, fh, fl, v, EH0, EL0, iH0, iL0, s_sat, kp, ks, T]
sym = tuple(list1 + list2)

## Now we define the RK2 Equations:
# =============================================================================
# u1 is the discretised function for s8 (time-dependent) and Jacobian Elements 
# =============================================================================
k1_1 = h*((((ns8*Ms8*iH0*ar)/(ne*F))*((exp(2*F*(Vg-EH0)/(R*T))/sqrt(fh*s8g))*(s4g)\
    - (exp(2*F*(EH0-Vg)/(R*T))/s4g)*(sqrt(fh*s8g))))\
        - ks*s8g)

u1 = k1_1 - s8g + s8_arr

du1ds8 = diff(u1, s8g)
du1ds8 = lambdify(sym, du1ds8, 'numpy')

du1ds4 = diff(u1, s4g)
du1ds4 = lambdify(sym, du1ds4, 'numpy')

du1dV = diff(u1, Vg)
du1dV = lambdify(sym, du1dV, 'numpy')

u1 = lambdify(sym, u1, 'numpy')

# =============================================================================
# u2 is the discretised function for s4 (time-dependent) and Jacobian Elements
# =============================================================================
k2_1 = h*(((-(ns8*Ms8*iH0*ar)/(ne*F))*((exp(2*F*(Vg-EH0)/(R*T))/sqrt(fh*s8g))*(s4g)\
    - (exp(2*F*(EH0-Vg)/(R*T))/s4g)*(sqrt(fh*s8g))))\
        + ks*s8g + (((ns4*Ms8*iL0*ar)/(ne*F))*((exp(2*F*(Vg-EL0)/(R*T))/sqrt(fl*s4g))*(sqrt(s2g*(sg**2)))\
            - (exp(2*F*(EL0-Vg)/(R*T))/(sqrt(s2g*(sg**2))))*(sqrt(fl*s4g)))))

u2 = k2_1 - s4g + s4_arr

du2ds8 = diff(u2, s8g)
du2ds8 = lambdify(sym, du2ds8, 'numpy')

du2ds4 = diff(u2, s4g)
du2ds4 = lambdify(sym, du2ds4, 'numpy')

du2ds2 = diff(u2, s2g)
du2ds2 = lambdify(sym, du2ds2, 'numpy')

du2ds = diff(u2, sg)
du2ds = lambdify(sym, du2ds, 'numpy')

du2dV = diff(u2, Vg)
du2dV = lambdify(sym, du2dV, 'numpy')

u2 = lambdify(sym, u2, 'numpy')

# =============================================================================
# u3 is the discretised function for s2 (time-dependent) and Jacobian Elements  
# =============================================================================
k3_1 = h*(((-(ns2*Ms8*iL0*ar)/(ne*F))*((exp(2*F*(Vg-EL0)/(R*T))/sqrt(fl*s4g))*(sqrt(s2g*(sg**2)))\
    - (exp(2*F*(EL0-Vg)/(R*T))/(sqrt(s2g*(sg**2))))*(sqrt(fl*s4g)))))

u3 = k3_1 - s2g + s2_arr

du3ds4 = diff(u3, s4g)
du3ds4 = lambdify(sym, du3ds4, 'numpy')

du3ds2 = diff(u3, s2g)
du3ds2 = lambdify(sym, du3ds2, 'numpy')

du3ds = diff(u3, sg)
du3ds = lambdify(sym, du3ds, 'numpy')

du3dV = diff(u3, Vg)
du3dV = lambdify(sym, du3dV, 'numpy')

u3 = lambdify(sym, u3, 'numpy')

# =============================================================================
# u4 is the discretised function for s (time-dependent) and Jacobian Elements    
# =============================================================================
k4_1 = h*((-2*((ns*Ms8*iL0*ar)/(ne*F))*((exp(2*F*(Vg-EL0)/(R*T))/sqrt(fl*s4g))*(sqrt(s2g*(sg**2)))\
    - (exp(2*F*(EL0-Vg)/(R*T))/(sqrt(s2g*(sg**2))))*(sqrt(fl*s4g))))\
        - ((kp*spg)/(v*ps))*(sg - s_sat))

u4 = k4_1 - sg + s_arr

du4ds4 = diff(u4, s4g)
du4ds4 = lambdify(sym, du4ds4, 'numpy')

du4ds2 = diff(u4, s2g)
du4ds2 = lambdify(sym, du4ds2, 'numpy')

du4ds = diff(u4, sg)
du4ds = lambdify(sym, du4ds, 'numpy')

du4dsp = diff(u4, spg)
du4dsp = lambdify(sym, du4dsp, 'numpy')

du4dV = diff(u4, Vg)
du4dV = lambdify(sym, du4dV, 'numpy')

u4 = lambdify(sym, u4, 'numpy')

# =============================================================================
# u5 is the discretised function for I = iH + iL (Non-time-dependent) and Jacobian Elements   
# =============================================================================
## This does not require k1 and k2 as it is not a time dependent equation
u5 = I + iH0*ar*((exp(2*F*(Vg-EH0)/(R*T))/sqrt(fh*s8g))*(s4g)\
                  - (exp(2*F*(EH0-Vg)/(R*T))/s4g)*(sqrt(fh*s8g)))\
    + iL0*ar*((exp(2*F*(Vg-EL0)/(R*T))/sqrt(fl*s4g))*(sqrt(s2g*(sg**2)))\
               - (exp(2*F*(EL0-Vg)/(R*T))/(sqrt(s2g*(sg**2))))*(sqrt(fl*s4g)))

du5ds8 = diff(u5, s8g)
du5ds8 = lambdify(sym, du5ds8, 'numpy')

du5ds4 = diff(u5, s4g)
du5ds4 = lambdify(sym, du5ds4, 'numpy')

du5ds2 = diff(u5, s2g)
du5ds2 = lambdify(sym, du5ds2, 'numpy')

du5ds = diff(u5, sg)
du5ds = lambdify(sym, du5ds, 'numpy')

du5dV = diff(u5, Vg)
du5dV = lambdify(sym, du5dV, 'numpy')

u5 = lambdify(sym, u5, 'numpy')

# =============================================================================
# u6 is the discretised function for sp (time-dependent) (sp is Precipitated Sulfur) and Jacobian Elements
# =============================================================================
k5_1 = ((h*kp*spg)/(v*ps))*(sg - s_sat)

u6 = k5_1 - spg + sp_arr

du6ds = diff(u6, sg)
du6ds = lambdify(sym, du6ds, 'numpy')

du6dsp = diff(u6, spg)
du6dsp = lambdify(sym, du6dsp, 'numpy')

u6 = lambdify(sym, u6, 'numpy')
