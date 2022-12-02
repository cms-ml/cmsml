import numpy as np

def count_parameter(model, verbose=True):
    number_of_parameter = np.sum(
        [np.prod(np.array(p.shape.as_list())) for p in model.trainable_variables])
    if verbose:
        print(f'Model has {number_of_parameter} parameter')
    return number_of_parameter
