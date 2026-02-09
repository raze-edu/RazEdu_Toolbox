# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport exp, tanh as c_tanh
from cython.view cimport array as cvarray
import array

cpdef double dot_product(double[:] inputs, double[:] weights) nogil:
    """
    Calculates the dot product of two memoryviews (inputs and weights).
    Releases the GIL for multithreading support.
    """
    cdef double result = 0.0
    cdef int i
    cdef int n = inputs.shape[0]
    
    # Simple check, though calling code should ensure this for max speed
    if weights.shape[0] != n:
        with gil:
            raise ValueError("Inputs and weights must have the same length")
        
    for i in range(n):
        result += inputs[i] * weights[i]
        
    return result

cpdef double sigmoid(double x) nogil:
    """
    Sigmoid activation function.
    """
    if x < -709.0:
        return 0.0
    if x > 709.0:
        return 1.0
    return 1.0 / (1.0 + exp(-x))

cpdef double relu(double x) nogil:
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    return x if x > 0.0 else 0.0

cpdef double tanh(double x) nogil:
    """
    Hyperbolic tangent activation function.
    """
    return c_tanh(x)

cpdef list softmax(double[:] x):
    """
    Softmax activation function.
    Returns a list of probabilities.
    Input must be a memoryview (e.g. array.array or numpy array).
    """
    cdef double sum_exp = 0.0
    cdef double val
    cdef double e_val
    cdef int i
    cdef int n = x.shape[0]
    # We need to store intermediates. 
    # Since we return a python list, we can't easily do this purely in nogil 
    # without managing memory manually.
    # For now, we will keep the main logic here but we can parallelize parts if needed.
    # Softmax is usually per-layer, so individual node ops are less common to thread *within* the function,
    # but the function itself can be called in a thread.
    
    cdef list exp_values = []
    
    # Calculate exponentials and sum
    for i in range(n):
        val = x[i]
        e_val = exp(val)
        exp_values.append(e_val)
        sum_exp += e_val
        
    # Normalize
    return [val / sum_exp for val in exp_values]
