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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# array adt

"""
**Dynamic Array**: collection of elements of type E in linear order
A contiguous block of memory that resizes automatically when it runs out of space.

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


# Generic Type
T = TypeVar("T")


# interface
class SequenceADT(ABC, Generic[T]):
    """Sequence ADT: models an ordered, finite collection of elements, each accessible by an integer position (index)."""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def get(self, index) -> T:
        """Return element at index i"""
        pass

    @abstractmethod
    def set(self, index, value: T):
        """Replace element at index i with x"""
        pass

    @abstractmethod
    def insert(self, index, value: T):
        """Insert x at index i, shift elements right"""
        pass

    @abstractmethod
    def delete(self, index: int) -> T:
        """Remove element at index i, shift elements left"""
        pass

    @abstractmethod
    def append(self, value: T):
        """Add x at end N-1"""
        pass

    @abstractmethod
    def prepend(self, value: T):
        """Insert x at index 0"""
        pass

    @abstractmethod
    def index_of(self, value: T) -> Optional[int]:
        """Return index of first x (if exists)"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def __len__(self) -> int:
        """Return number of elements - formally defined as size()"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """returns true if sequence is empty"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """removes all items from the sequence"""
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        """True if x exists in sequence"""
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        pass


CTYPES_DATATYPES = {
    int: ctypes.c_int,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    str: ctypes.py_object,  # arbitrary Python object (strings)
    object: ctypes.py_object,  # any Python object
}

NUMPY_DATATYPES = {
    int: numpy.int32,
    float: numpy.float64,
    bool: numpy.bool_,
}


class VectorArray(SequenceADT[T]):
    """Dynamic Array — automatically resizes as elements are added."""
    def __init__(self, capacity: int, datatype: type, datatype_map: dict = CTYPES_DATATYPES) -> None:

        # datatype
        self.datatype = datatype
        self.datatype_map = datatype_map
        self.type = None

        # Core Array Properties
        self.min_capacity = max(4, capacity)  # sets a minimum size for the array
        self.capacity = capacity  # sets the total amount of spaces for the array
        self.size = 0  # tracks the number of elements in the array
        self.array = self._initialize_new_array(self.capacity)  # creates a new ctypes/numpy array with a specified capacity


    # ----- Utility -----
    def _initialize_new_array(self, capacity):
        """chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)"""
        if self.datatype_map == CTYPES_DATATYPES:
            new_array = self._init_ctypes_array(capacity)
            return new_array
        elif self.datatype_map == NUMPY_DATATYPES:
            new_array = self._init_numpy_array(capacity)
            return new_array
        else:
            raise ValueError(f"Datatype Map Unknown... Map: {self.datatype_map}")

    def _init_ctypes_array(self, capacity):
        """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
        # setting ctypes datatype -- needed for the array. (object is a general all purpose datatype)
        if self.datatype not in self.datatype_map:
            self.type = ctypes.py_object    # general all purpose python object
        else:
            self.type = CTYPES_DATATYPES[self.datatype]  # maps type of array to ctype
        # creates a class object - an array of specified number of a specified type
        dynamic_array_cls = self.type * capacity
        # initializes array with preallocated memory block
        new_ctypes_array = dynamic_array_cls()
        return new_ctypes_array

    def _init_numpy_array(self, capacity):
        """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
        if self.datatype not in NUMPY_DATATYPES:
            self.type = object  # general all purpose python object.
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        return new_numpy_array

    def _enforce_type(self, value):
        """type enforcement - checks that the value matches the prescribed datatype."""
        if not isinstance(value, self.datatype):
            raise TypeError(f"Error: Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}")

    def _index_boundary_check(self, index, is_insert: bool = False):
        """
        Checks that the index is a valid number for the array. 
        index needs to be greater than 0 and smaller than the number of elements (size)
        """
        if is_insert:
            if index < 0 or index > self.size:
                raise IndexError("Error: Index is out of bounds.")
        else:
            if index < 0 or index >= self.size:
                raise IndexError("Error: Index is out of bounds.")

    def _grow_array(self):
        """
        Step 1: Store existing array data and capacity.
        Step 2: Initialize new array with * 2 capacity
        Step 3: Copy old items to new array
        Step 4: Update the capacity to reflect the new extended capacity.
        Step 5: return the array for use in the program.
        """
        old_array = self.array
        old_capacity = self.capacity
        new_capacity = self.capacity * 2
        new_array = self._initialize_new_array(new_capacity)
        for i in range(self.size):
            new_array[i] = old_array[i]
        self.capacity = new_capacity
        return new_array

    def _shrink_array(self):
        """Shrink array in half"""
        old_array = self.array
        old_capacity = self.capacity
        new_capacity = max(self.min_capacity, self.capacity // 2)
        new_array = self._initialize_new_array(new_capacity)
        for i in range(self.size):
            new_array[i] = old_array[i]
        self.capacity = new_capacity
        return new_array

    def __str__(self) -> str:
        """a list of strings representing all the elements in the array"""
        items = ", ".join(str(self.array[i]) for i in range(self.size))
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        return f"Array Items: {items}: Capacity: {self.size}/{self.capacity} Type: {string_datatype}"


    # ----- Canonical ADT Operations -----
    def get(self, index):
        """Return element at index i"""
        self._index_boundary_check(index)
        result = self.array[index]
        return cast(T, result)

    def set(self, index, value):
        """Replace element at index i with x"""
        self._enforce_type(value)
        self._index_boundary_check(index)
        self.array[index] = value

    def insert(self, index, value):
        """
        Insert x at index i, shift elements right:
        Step 1: Loop through elements: Start at the end & go backwards. Stop at the index element (where we want to insert.)
        Step 2: copy element from the previous index (the left) - this shifts every element to the right.
        Step 3: Now the target index will contain a duplicate value - which we will overwrite with the new value
        Step 4: Increment Array Size Tracker
        """

        self._enforce_type(value)
        self._index_boundary_check(index, is_insert=True)
        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array()

        # if index value is the end of the array - utilize O(1) append
        if index == self.size:
            self.append(value)
            return

        # move all array elements right.
        for i in range(self.size, index, -1): 
            self.array[i] = self.array[i-1]   # (e.g. elem_4 = elem_3)
        self.array[index] = value 
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
            raise ValueError("Error: Array is Empty.")
        
        self._index_boundary_check(index)

        # dynamically shrink array if capacity at 25% and greater than min capacity
        if self.size == self.capacity // 4 and self.capacity > self.min_capacity:
            self.array = self._shrink_array()

        deleted_value = self.array[index]   # store index for return
        # shift elements left -- Starts from the deleted index (Goes Backwards)
        for i in range(index, self.size - 1):  
            self.array[i] = self.array[i + 1]  # (elem4 = elem5)
        if self.type is object or self.type is ctypes.py_object:
            self.array[self.size - 1] = None   # removes item from the end of the stored items 
        self.size -= 1  # decrement size tracker
        return deleted_value

    def append(self, value):
        """Add x at end -- O(1)"""
        self._enforce_type(value)

        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array()

        self.array[self.size] = value
        self.size += 1

    def prepend(self, value):
        """Insert x at index 0 -- O(N) - Same logic as insert, shift elems right"""
        self._enforce_type(value)
        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array() 

        for i in range(self.size, 0, -1):
            self.array[i] = self.array[i-1]

        self.array[0] = value
        self.size += 1

    def index_of(self, value):
        """Return index of first x (if exists)"""
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
        self.array = self._initialize_new_array(self.min_capacity)
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

def main():
    # test data initialization
    print("=== VectorArray Full Test ===")

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

    def run_array_tests(datatype: type, test_values: list, datatype_map: dict = CTYPES_DATATYPES):
        print(f"=== Testing {datatype.__name__} array===")

        # create array with minimum capacity 6 or length of test data
        min_capacity = max(6, len(test_values))
        arr = VectorArray[datatype](min_capacity, datatype, datatype_map)

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
        except TypeError as e:
            print(f"Caught expected type error: {e}")

        # --- Index errors ---
        try:
            arr.get(999)
        except IndexError as e:
            print(f"Caught expected index error: {e}")

        # --- Empty array delete ---
        arr.clear()
        try:
            arr.delete(0)
        except (IndexError, ValueError) as e:
            print(f"Caught expected error on deleting from empty array: {e}")

        # --- Dynamic growth test ---
        print("\nDynamic Growth Test")
        print(f"{arr}")
        for i in range(len(test_values)*2):  # trigger growth
            arr.append(test_values[i % len(test_values)])
        print(f"{arr}")

        # --- Dynamic shrink test ---
        print("\nDynamic Shrink Test")
        print(f"{arr}")
        while len(arr) > 2:  # deleting to trigger shrink
            removed = arr.delete(0)
        print(f"{arr}")

        # --- Iteration test ---
        print("\nIteration test:")
        for item in arr:
            print(f"Iterated item: {item}")

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
