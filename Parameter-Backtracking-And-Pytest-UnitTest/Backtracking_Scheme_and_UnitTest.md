## Script/File Explanations ##
* ```func.py```
  - This file contains the function definitions for all the Backwards Euler formulations and partial derivatives.
* ```LiS_Backtrack_Solver.py```
  - This script contains the solver function and model class that can be accessed as a Python module.
* ```module_func.py```
  - This script contains 2 functions namely the ```labels()``` function which creates a dictionary to allow access to each cycle charge and discharge variable values     and the ```concatenate()``` function which allows the user to combine values for a single variable over the whole microcycling period.
* ```Test_Backtrack.py```
  - This script is used to run the solver function by importing it and defining the initial conditions.
* ```Test_Microcycling_v2.0.py```
  - This is a similar script to the above, however is used for microcycling.
* ```Test_Saved_Microcycling.py```
  - This script is used to access the saved arrays in form of a npz file, plotting of the microcycling process is also defined in this script.
* ```pytest_lis_solver.py```
  - This script uses the ```pytest``` module to test stability of the solver with different parameterization. 
