from LiS_Backtrack_Solver import LiS_Solver
import numpy as np
import warnings
import pytest

## Now we define a function that calls and LiS solver ##
## The function will take the input in format of the upd_params dictionary ##

def LiS_Solver_Method(upd_param, params_backtracked):
    ## Now we call the solver and solve ##
    t_end = 7200 # End time
    h0 = 0.5 # Initial step size

    ## Initialize the variable values and arrays
    s8i = 2.6892000000000003
    s4i = 0.0027
    s2i = 0.002697299116926997
    si = 8.83072852310722e-10
    spi = 2.7e-06
    Vi = 2.430277479547109
    I = 1.7 # Current (constant)
    break_voltage = 1.98 # Voltage point where simulation is stopped to prevent Singular Matrix Error

    ## Run the solver and save results within npz file
    solved = LiS_Solver(s8i, s4i, s2i, si, Vi, spi, t_end, h0, I, break_voltage, state='Discharge', 
                        params_backtrack=params_backtracked, upd_params=upd_param)
    
    return(solved)

## Define the pytest inputs using decorator: ##
@pytest.mark.parametrize("inputs", [
    ## inputs[0]: is the updated parameter
    ## inputs[1]: is the parameter for backtracking
    [{"EL0": 2.195, "v": 0.00114}, {}],
    [{"EL0": 2.1}, {"EL0": 2.15}],
    [{"EL0": 2.0}, {"EL0": 2.1}],
    [{"EL0": 1.95}, {"EL0": 2.1}],
    [{"EL0": 1.8}, {"EL0": 1.85}]

])

## Define test function ##
def test_LiS_Solver(inputs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        LiS_Solver_Method(inputs[0], inputs[1])
        
        if any(issubclass(warn.category, RuntimeWarning) for warn in w):
            pytest.fail("Overflow warning encountered during calculations for input")
        
        if any(isinstance(warn.message, np.linalg.LinAlgError) for warn in w):
            pytest.fail("Singular matrix error encountered during calculations for input")

# Run the test function using pytest
pytest.main([__file__])
        