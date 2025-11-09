# region standard imports

from typing import (
    Generic,
    TypeVar,
    List,
    Dict,
    Optional,
    Callable,
    Any,
    cast,
    Iterator,
    Generator,
    Iterable,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes

# endregion

# region custom imports
from utils.array_constants import CTYPES_DATATYPES, NUMPY_DATATYPES, ARRAY_GROWTH_FACTOR, ARRAY_SHRINK_FACTOR
from utils.custom_types import T

if TYPE_CHECKING:   # does not run at runtime - avoids circular imports.
    from ds.primitives.arrays.dynamic_array import VectorArray

# endregion


def index_boundary_check(index: int, capacity: int, is_insert: bool = False) -> None:
    """Checks that the index is a valid number for the array. -- index needs to be greater or equal to 0 and smaller than the number of elements (size)"""
    if is_insert:
        if index < 0 or index > capacity:
            raise IndexError("Error: Index is out of bounds.")
    else:
        if index < 0 or index >= capacity:
            raise IndexError("Error: Index is out of bounds.")

def init_ctypes_array(datatype: type, capacity: int) -> ctypes.Array:
    """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
    # setting ctypes datatype -- needed for the array. (object is a general all purpose datatype)
    if datatype not in CTYPES_DATATYPES:
        ctypes_datatype = ctypes.py_object  # general all purpose python object
    else:
        ctypes_datatype = CTYPES_DATATYPES[datatype]  # maps type of array to ctype
    # creates a class object - an array of specified number of a specified type
    dynamic_array_cls = ctypes_datatype * capacity
    # initializes array with preallocated memory block
    new_ctypes_array = dynamic_array_cls()
    return new_ctypes_array

def init_numpy_array(datatype: type, capacity: int) -> numpy.ndarray:
    """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
    if datatype not in NUMPY_DATATYPES:
        numpy_datatype = NUMPY_DATATYPES.get(datatype, object)  # general all purpose python object.
        new_numpy_array = numpy.empty(capacity, dtype=numpy_datatype)
    else:
        numpy_datatype = NUMPY_DATATYPES[datatype]
        new_numpy_array = numpy.empty(capacity, dtype=numpy_datatype)
    return new_numpy_array

def initialize_new_array(datatype: type, capacity: int, datatype_map: dict = CTYPES_DATATYPES) -> ctypes.Array | numpy.ndarray:
    """chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)"""
    if datatype_map == CTYPES_DATATYPES:
        new_array = init_ctypes_array(datatype, capacity)
        return new_array
    elif datatype_map == NUMPY_DATATYPES:
        new_array = init_numpy_array(datatype, capacity)
        return new_array
    else:
        raise ValueError(f"Error: Datatype Map Unknown... Map: {datatype_map}")

def grow_array(array_obj: "VectorArray", resize_factor: int = ARRAY_GROWTH_FACTOR)  -> ctypes.Array | numpy.ndarray:
    """
    Grows the array by a predetermined factor (takes the class instance as a parameter. - self.)
    Step 1: Store existing array data and capacity.
    Step 2: Initialize new array with * 2 capacity
    Step 3: Copy old items to new array
    Step 4: Update the capacity to reflect the new extended capacity.
    Step 5: return the array for use in the program.
    """
    old_array = array_obj.array
    old_capacity = array_obj.capacity
    new_capacity = old_capacity * resize_factor
    new_array = initialize_new_array(array_obj.datatype, new_capacity, array_obj.datatype_map)
    for i in range(array_obj.size):
        new_array[i] = old_array[i]
    array_obj.capacity = new_capacity
    return new_array

def shrink_array(array_obj: "VectorArray", resize_factor: int = ARRAY_SHRINK_FACTOR)  -> ctypes.Array | numpy.ndarray:
    """Shrink array by resize factor - takes the class instance as a parameter. - self"""
    old_array = array_obj.array
    old_capacity = array_obj.capacity
    new_capacity = max(array_obj.min_capacity, old_capacity // resize_factor)
    new_array = initialize_new_array(array_obj.datatype, new_capacity, array_obj.datatype_map)
    for i in range(array_obj.size):
        new_array[i] = old_array[i]
    array_obj.capacity = new_capacity
    return new_array

def shift_elements_right(array_obj: "VectorArray", index: int, value: T) -> None:
    """move all array elements right - aka insert -- O(N)"""
    for i in range(array_obj.size, index, -1):
        array_obj.array[i] = array_obj.array[i - 1]  # (e.g. elem_4 = elem_3)
    array_obj.array[index] = value

def shift_elements_left(array_obj: "VectorArray", index: int) -> None:
    """shift elements left -- Starts from the deleted index (Goes Backwards) -- aka delete -- O(N)"""
    for i in range(index, array_obj.size - 1):  
        array_obj.array[i] = array_obj.array[i + 1]  # (elem4 = elem5)

    # looks through datatype map to see specific type that the array is using
    # (can be a special ctype or numpy type. Defaults to ctypes.py_object - which aligns 100% with a python object.)
    specific_type = array_obj.datatype_map.get(array_obj.datatype, ctypes.py_object)
    # objects need to be dereferenced. (numbers dont)
    if specific_type is object or specific_type is ctypes.py_object:
        array_obj.array[array_obj.size - 1] = None   # removes item from the end of the stored items 
