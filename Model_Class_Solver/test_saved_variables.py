import numpy as np
import matplotlib.pyplot as plt

# Load the variable arrays from the file
data = np.load('variable_arrays.npz')
solved = data['solved']
I = data['I']

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
EH0 = 2.35
EL0= 2.195
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

# Now we plot:
t = np.array(t)
Ah = (t/3600)*I
plt.plot(Ah,V)
plt.title(f"Cell Output Voltage vs Discharge Capacity {I}A")
plt.ylabel("Potential (V)")
plt.xlabel("Discharge Capacity (Ah)")
plt.xlim([-0.25,3.5])
plt.ylim([2.2,2.45])
#plt.savefig('1.7A_Voltage_Output.png', dpi=1200)
plt.show()

# =============================================================================
# plt.plot(Ah,iH, c= 'green', label= 'iH')
# plt.ylabel("iH (A)")
# plt.xlabel("Discharge Capacity (Ah)")
# 
# plt.plot(Ah,iL, c = 'red', label = 'iL')
# plt.ylabel("iL (A)")
# plt.xlabel("Discharge Capacity (Ah)")
# plt.ylim([-0.5,2])
# plt.legend()
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
