## SCRIPT DESCRIPTIONS HERE ##
```func.py```
  - This script contains the defined symbols and discretized equations, including some helper functions to define the function arrays and jacobian matrix.

```LiS_Backtrack_Solver.py```
  - This script contains the ```LiS_Model()``` calss variable and the solver function ```LiS_Solver()```, the only change that needs to be made here is to include all the defined/changed parameters into the ```__init()``` function of the ```LiS_Model()``` class.

```module_func.py```
  - This script contains 2 functions namely the ```labels()``` function which creates a dictionary to allow access to each cycle charge and discharge variable values and the ```concatenate()``` function which allows the user to combine values for a single variable over the whole microcycling period.

```Test_Backtrack.py```
  - This script calls the ```LiS_Solver()``` function from the ```LiS_Backtrack_Solver.py``` script to run a simulation by defining the initial values and other arguments.

```Test_Microcycling_v3.py```
  - This script runs the solver for micro-cycling by defining discharge and charge states and saves the overall output into a npz file.

```Test_Saved_Microcycling.py```
  - This script loads the saved data from the previous script and calls the functions from the ```module_func.py``` script to process and plot the data.

```Test_General_GD_ADAM.py```
  - This script contains the functions required for the gradient descent scheme, refer to the user manual for further information.

```Test_Gradient_Descent.py```
  - This is a test script for the gradient descent involving interpolating the simulation and experimental data.

```vol_data```
  - Pickled file containing the experimental voltage and capacity data.
