import numpy as np

# =============================================================================
# This Script Contains the lables function to create a dictionary and 
# the concatenate function to concatenate the whole cycling variable
# =============================================================================

## Define Dictionary function to store cylce values ##
def labels(my_list):
    labels = {}  # Empty dictionary to store labels

    # Iterate over the cycles in the list
    for i, cycle in enumerate(my_list, start=1):
        cycle_label = f"cycle{i}"
        cycle_dict = {}  # Dictionary to store discharge and charge labels
    
        # Assign discharge and charge labels to the two lists
        cycle_dict["discharge"] = cycle[0]
        cycle_dict["charge"] = cycle[1]
    
        labels[cycle_label] = cycle_dict
    return(labels)

## Define concatenate function to allow to merge discharge and charge for cycle variables
def concatenate(labels, var):
    var_list = []
    for i in range(len(labels)):
        discharge = labels[f"cycle{i+1}"]["discharge"]
        charge = labels[f"cycle{i+1}"]["charge"]
        var_list.append(discharge[var])
        var_list.append(charge[var])

    return(np.concatenate(var_list))