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
    Iterable
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence
# endregion


# region custom imports
from utils.constants import CTYPES_DATATYPES, NUMPY_DATATYPES, ARRAY_MIN_CAPACITY, SHRINK_CAPACITY_RATIO
from user_defined_types.generic_types import (
    T,
    K,
    iKey,
    ValidDatatype,
    ValidIndex,
    Index,
    TypeSafeElement,
)
from utils.validation_utils import DsValidation
from utils.representations import ArrayRepr, ViewRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.array_utils import ArrayUtils

# endregion


"""
Dynamic Array: 

collection of elements of type E in linear order
A contiguous block of memory that resizes automatically when it runs out of space.

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


# ? potential features to add.
# ? Safe and Unsafe modes: Safe = Type Safety, Unsafe = No rules.
# ? reversed iteration
# ? step iteration
# ! batch operations, (get, set, insert, append, replace etc)
# ? immutable / read only views.
# ? read only array.... maybe....
# ? traverse method for array.


class VectorView(Generic[T]):
    """Internal class to represent a view of a Vector. Similar to a python slice, but without copying the values (expensive) -- view is O(1), python slice is O(N)"""
    def __init__(self, datatype: type, array: Any, start: int = 0, length: Optional[int] = None, stride: int = 1) -> None:
        self._view = array  # the original data array, shared with the view.
        self._start = start # start of the view.
        self._length = length if length is not None else len(array) - start  # length of view
        self._stride = stride   # step value
        self._datatype = ValidDatatype(datatype)

        # composed objects
        self._utils: ArrayUtils = ArrayUtils(self)
        self._validators: DsValidation = DsValidation()
        self._desc: ViewRepr = ViewRepr(self)

    @property
    def datatype(self):
        return self._datatype

    def __len__(self):
        return self._length

    def __getitem__(self, index: int | slice) -> T | "VectorView[T]":
        """retrieves an item from the view."""
        if isinstance(index, slice):
            # inbuilt method for slice (.indices()) - Converts None to actual numbers. Converts negative indices to positive. Makes sure slicing stays within bounds of _length.
            start, stop, step = index.indices(self._length) 
            view_start_index = self._start + start * self._stride
            view_length = (stop - start + (step - 1)) // step
            view_stride = self._stride * step
            return VectorView(self._datatype, self._view, view_start_index, view_length, view_stride)
        index = ValidIndex(index, self._length, array_insert=True)
        return self._view[self._start + index * self._stride]  # access index

    def __setitem__(self, index: int, value: Any):
        """replaces a value of the view."""
        index = ValidIndex(index, self._length, array_insert=True)
        value = TypeSafeElement(value, self.datatype)
        self._view[self._start + index * self._stride] = value

    def __iter__(self) -> Generator[Any , None, None]:
        """iterates through view elements"""
        for i in range(self._length):
            yield self._view[self._start + i * self._stride]

    def __str__(self) -> str:
        return self._desc.str_view()

    def __repr__(self) -> str:
        return self._desc.repr_view()


class VectorArray(SequenceADT[T], CollectionADT[T]):
    """Dynamic Array — automatically resizes as elements are added."""
    def __init__(self, capacity: int, datatype: type, datatype_map: dict = CTYPES_DATATYPES, is_static: bool = False) -> None:
        # composed objects
        self._utils: ArrayUtils = ArrayUtils(self)
        self._validators: DsValidation = DsValidation()
        self._desc: ArrayRepr = ArrayRepr(self)

        # datatype
        self.datatype = ValidDatatype(datatype)
        self.datatype_map = datatype_map

        # Core Array Properties
        self.min_capacity = max(ARRAY_MIN_CAPACITY, capacity)  # min size for array
        self.capacity = capacity  # sets total amount of spaces for the array (# todo same change to private)
        self.size = 0  # tracks number of elements in the array (# todo change to protected with property)
        # creates a new ctypes/numpy array with a specified capacity
        self.array = self._utils.initialize_new_array(self.datatype, self.capacity, self.datatype_map)
        self._is_static = is_static

    # ----- Utility -----

    @property
    def is_static(self):
        return self._is_static

    def __str__(self) -> str:
        """a list of strings representing all the elements in the array"""
        return self._desc.str_array()

    def __repr__(self) -> str:
        """ returns memory address and info"""
        return self._desc.repr_array()

    def __getitem__(self, index: Index | slice) -> T | VectorView:
        """Built in override - adds indexing, & slicing but for views instead of copies (like python slice)"""
        # convert python slice parameters to view logic and return a view obj instance.
        if isinstance(index, slice):
            view = self.array
            slice_start = index.start or 0
            view_length = (index.stop - (index.start or 0)) if index.stop is not None else None
            slice_step = index.step or 1
            return VectorView(self.datatype, view, slice_start, view_length, slice_step)
        valid_index = ValidIndex(index, self.capacity, array_insert=False)
        return self.get(valid_index)

    def __setitem__(self, index, value: T):
        """Built in override - adds indexing."""
        value = TypeSafeElement(value, self.datatype)
        self.set(index, value)

    # ----- Canonical ADT Operations -----
    def get(self, index):
        """Return element at index i"""
        index = ValidIndex(index, self.capacity, array_insert=False)
        result = self.array[index]
        return cast(T, result)

    def set(self, index, value):
        """Replace element at index i with x"""
        value = TypeSafeElement(value, self.datatype)
        self._validators.enforce_type(value, self.datatype)
        index = ValidIndex(index, self.capacity, array_insert=False)
        self.array[index] = value

    def insert(self, index, value):
        """
        Insert x at index i, shift elements right:
        Step 1: Loop through elements: Start at the end & go backwards. Stop at the index element (where we want to insert.)
        Step 2: copy element from the previous index (the left) - this shifts every element to the right.
        Step 3: Now the target index will contain a duplicate value - which we will overwrite with the new value
        Step 4: Increment Array Size Tracker
        """
        value = TypeSafeElement(value, self.datatype)
        index = ValidIndex(index, self.capacity, array_insert=True)

        # dynamically resize the array if capacity full.
        if self.size == self.capacity and self._is_static == False:
            self.array = self._utils.grow_array()
        elif self.size == self.capacity and self._is_static == True:
            raise DsOverflowError(f"Error: Array is currently at max capacity. {self.size}/{self.capacity}")

        # if index value is the end of the array - utilize O(1) append
        if index == self.size:
            self.append(value)
            return

        # move all array elements right.
        self._utils.shift_elements_right(index, value)

        self.size += 1  # update size tracker

    def delete(self, index):
        """
        Remove element at index i, shift elements left:
        Step 1: store index to return later (the deleted item)
        Step 2: Loop through elements from the index to the end of the array.
        Step 3: copy element from the future index (the right). This shifts each element left (from the target index point onwards.)
        Step 4: For the last element in the array, change value to None
        Step 5: decrement the size tracker.
        Step 6: return deleted value
        """

        if self.is_empty():
            raise DsUnderflowError("Error: Array is Empty.")

        index = ValidIndex(index, self.capacity, array_insert=False)

        # dynamically shrink array if capacity at 25% and greater than min capacity
        if self.size == self.capacity // SHRINK_CAPACITY_RATIO and self.capacity > self.min_capacity and self._is_static == False:
            self.array = self._utils.shrink_array()

        deleted_value = self.array[index]   # store index for return

        self._utils.shift_elements_left(index)
        self.size -= 1  # decrement size tracker

        return deleted_value

    def append(self, value):
        """Add x at end -- O(1)"""

        value = TypeSafeElement(value, self.datatype)

        # dynamically resize the array if capacity full.
        if self.size == self.capacity and self._is_static == False:
            self.array = self._utils.grow_array()
        elif self.size == self.capacity and self._is_static == True:
            raise DsOverflowError(f"Error: Array is currently at max capacity. {self.size}/{self.capacity}")

        self.array[self.size] = value
        self.size += 1

    def append_many(self, list_of_values: Iterable):
        """appends multiple values to the end of the array. works similar to python implementation"""
        for i in list_of_values:
            self.append(i)

    def prepend(self, value):
        """Insert x at index 0 -- O(N) - Same logic as insert, shift elems right"""

        value = TypeSafeElement(value, self.datatype)

        # dynamically resize the array if capacity full.
        if self.size == self.capacity and self._is_static == False:
            self.array = self._utils.grow_array()
        elif self.size == self.capacity and self._is_static == True:
            raise DsOverflowError(f"Error: Array is currently at max capacity. {self.size}/{self.capacity}")

        self._utils.shift_elements_right(0, value)
        self.size += 1

    def index_of(self, value):
        """Return index of first x (if exists)"""
        value = TypeSafeElement(value, self.datatype)
        for i in range(self.size):
            if self.array[i] == value:
                return i
        return None

    # ----- Meta Collection ADT Operations -----
    def __len__(self):
        """Return number of elements"""
        return self.size

    def is_empty(self):
        """returns true if sequence is empty"""
        return self.size == 0

    def clear(self):
        """removes all items and reinitializes a new array with the original capacity, resets the size tracker also"""
        self.array = self._utils.initialize_new_array(self.datatype, self.min_capacity, self.datatype_map)
        self.capacity = self.min_capacity
        self.size = 0

    def __contains__(self, value):
        """True if x exists in sequence"""
        for i in range(self.size):
            if self.array[i] == value:
                return True
        return False

    def __iter__(self):
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        for i in range(self.size):
            result = self.array[i]
            yield cast(T, result)


# Main -- Client Facing Code


def run_array_tests(
    datatype: type, 
    test_values: list, 
    datatype_map: dict = CTYPES_DATATYPES,
    ):
    print(f"=== Testing {datatype.__name__} array===")

    AI = type(
        "ArtificialPerson",
        (),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"NotAPerson({self.name})",
            "__repr__": lambda self: f"NotAPerson({self.name})",
        },
    )

    artificial = [AI(f"NotAPerson{i}") for i in range(6)]

    # create array with minimum capacity 6 or length of test data
    min_capacity = max(6, len(test_values))
    arr = VectorArray[datatype](min_capacity, datatype, datatype_map, is_static=False)

    print(f"Initial array: {arr}")

    # --- Core operations ---
    # append()
    for val in test_values:
        arr.append(val)
    print(f"After appends: {arr}")

    # prepened()
    arr.prepend(test_values[0])
    print(f"After prepend {test_values[0]}: {arr}")

    if len(test_values) > 2:
        # insert()
        arr.insert(2, test_values[1])
        print(f"Insert {test_values[1]} at index 2: {arr}")

        # set()
        arr.set(2, test_values[2])
        print(f"Set index 2 to {test_values[2]}: {arr}")

        # get()
        val = arr.get(2)
        print(f"Get index 2: expected {test_values[2]}, got {val}")

        # index_of()
        idx = arr.index_of(test_values[2])
        print(f"Index of {test_values[2]}: expected 2, got {idx}")

        # delete()
        deleted = arr.delete(2)
        print(f"Deleted index 2 (value {deleted}): {arr}")

    # --- Type enforcement ---
    try:
        arr.append(artificial[1])  # deliberately wrong
    except Exception as e:
        print(f"Caught expected type error: {e}")

    # --- Index errors ---
    try:
        arr.get(999)
    except Exception as e:
        print(f"Caught expected index error: {e}")

    # --- Empty array delete ---
    arr.clear()
    try:
        arr.delete(0)
    except (Exception, Exception) as e:
        print(f"Caught expected error on deleting from empty array: {e}")

    # --- Dynamic growth test ---
    print("\nDynamic Growth Test")
    print(f"{arr}")
    for i in range(len(test_values) * 2):  # trigger growth
        arr.append(test_values[i % len(test_values)])
    print(f"{arr}")

    # --- Dynamic shrink test ---
    print("\nDynamic Shrink Test")
    print(f"{arr}")
    while len(arr) > 2:  # deleting to trigger shrink
        removed = arr.delete(0)
    print(f"{arr}")

    print("\nre-adding items")
    print(f"{arr}")
    for i in range(len(test_values) * 2):  # trigger growth
        arr.append(test_values[i % len(test_values)])
    print(f"{arr}")

    # --- Iteration test ---
    print("\nIteration test:")
    half_array = len(arr) // 6
    subset = random.sample(list(arr), half_array)
    print(subset)
    for item in subset:
        print(f"Iterated item: {item}")
    print(f"\n{repr(arr)}\n")

    # testing View:
    new_view = arr[:6]
    print(new_view)
    print(repr(new_view))
    for item in new_view:
        print(f"Iterated item in View: {item}")


def main():
    # test data initialization
    print("=== VectorArray Full Test ===")

    # Input Data
    # Dynamic classes
    Person = type(
        "Person",
        (),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"Person({self.name})",
            "__repr__": lambda self: f"Person({self.name})",
        },
    )

    AI = type(
        "ArtificialPerson",
        (),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"NotAPerson({self.name})",
            "__repr__": lambda self: f"NotAPerson({self.name})",
        },
    )

    people = [Person(f"Person{i}") for i in range(6)]
    artificial = [AI(f"NotAPerson{i}") for i in range(6)]

    # Test data
    ints = [1,2,3,4,5,6]
    floats = [1.1,2.2,3.3,4.4,5.5,6.6]
    strings = [f"s{i}" for i in range(6)]
    bools = [True, False, True, False, True, False]
    lists = [[i] for i in range(6)]
    tuples = [(i,i+1) for i in range(6)]
    dicts = [{"key": i} for i in range(6)]

    


    # print(f"\ntesting CTYPES Array")
    # run_array_tests(int, ints, CTYPES_DATATYPES)
    # print(f"\ntesting NUMPY Array")
    # run_array_tests(int, ints, NUMPY_DATATYPES)
    # print(f"\ntesting CTYPES Array")
    # run_array_tests(float, floats, CTYPES_DATATYPES)
    # print(f"\ntesting NUMPY Array")
    # run_array_tests(float, floats, NUMPY_DATATYPES)

    # run_array_tests(str, strings)
    # run_array_tests(bool, bools)
    # run_array_tests(list, lists)
    # run_array_tests(tuple, tuples)
    # run_array_tests(dict, dicts)
    run_array_tests(Person, people)


if __name__ == "__main__":
    main()
