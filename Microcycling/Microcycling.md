* The LiS_Model_Solver.py script similar to the model class module.py script contains the Li-S model class and solver function definition.
* The module_func.py script contains 2 functions namely the labels function which creates a dictionary to allow access to each cycle charge and discharge variable values and the concatenate function which allows the user to combine values for a single variable over the whole microcycling period.
* The Test_Microcycling_v2.0.py script runs the solver for microcycling by defining discharge and charge states and saves the overall output into a npz file.
* The Test_Saved_Microcycling.py loads the saved data from the previous script and calls the functions from the module_func.py script to process and plot the data.
