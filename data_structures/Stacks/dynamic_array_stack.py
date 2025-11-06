from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator, Generator
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


"""
Dynamic array based Stack. Automatically resizes when it gets close to full.
Dynamic capacity (resize up/down, usually ×2 / ÷2).
Double capacity on full 
Half capacity when ≤25% full
user supplied initial capacity > 1
Static Type Validation
Overflow & Underflow Errors
"""


T = TypeVar("T")


# interface
class StackADT(ABC, Generic[T]):
    """Stack ADT - defines the necessary methods for a stack"""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def push(self, value: T) -> None:
        pass

    @abstractmethod
    def pop(self) -> T:
        pass

    @abstractmethod
    def peek(self) -> T:
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass


CTYPES_DATATYPES = {
    int: ctypes.c_int,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    str: ctypes.py_object,      # arbitrary Python object (strings)
    object: ctypes.py_object,   # any Python object
}

NUMPY_DATATYPES = {
    int: numpy.int32,
    float: numpy.float64,
    bool: numpy.bool_,
}


class DynamicArrayStack(StackADT[T]):
    """Dynamically sized array based stack. The array will resize itself when it gets close to full."""
    def __init__(self, initial_capacity: int, datatype: type, datatype_map: dict) -> None:

        # check initial capacity is greater than 1
        if initial_capacity < 1:
            raise ValueError("Initial Size of the Stack must be greater than 1")
        self.min_capacity = initial_capacity
        self.capacity = initial_capacity
        self.datatype = datatype
        self.datatype_map = datatype_map
        self.type = None
        self.dynamic_array = self._initialize_new_array(self.capacity)

        # tracks the next empty slot in the stack.
        self.top = 0

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
            self.type = ctypes.py_object
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
            self.type = object
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        return new_numpy_array

    def __str__(self):
        """a list of strings representing all the elements in the stack. Goes from Top to Bottom"""
        items = ", ".join(
            str(self.dynamic_array[i]) for i in range(self.top - 1, -1, -1)
        )
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        return f"Type: {string_datatype}: Capacity: {self.top}/{self.capacity} Stack Items: [Top]->> {items}"

    def _grow_stack(self):
        """initialize a new array with X2 Capacity, then copy the items over and return array"""
        new_capacity = self.capacity * 2
        new_array = self._initialize_new_array(new_capacity)
        # Step 2: Copy existing items to this new array.
        for i in range(self.top):
            new_array[i] = self.dynamic_array[i]
        # Step 3: update capacity to reflect new size.
        self.capacity = new_capacity
        return new_array

    def _shrink_stack(self):
        """Initialize a new array with /2 capacity (wont go lower than the initial array size), then copy the items over and return array"""
        new_capacity = max(self.min_capacity, self.capacity // 2)
        new_array = self._initialize_new_array(new_capacity)
        # Step 2: Copy existing items to this new array.
        for i in range(self.top):
            new_array[i] = self.dynamic_array[i]
        # Step 3: update capacity to reflect new size.
        self.capacity = new_capacity
        return new_array

    # ----- Canonical ADT Operations -----
    def push(self, value: T) -> None:
        """Adds a new item to the end of the stack."""
        # is stack over 75% or Under 25% capacity? - resize stack (this doubles as overflow check)
        if self.top >= self.capacity * 3 // 4:
            self.dynamic_array = self._grow_stack()
        elif self.top <= self.capacity // 4 and self.capacity > self.min_capacity:
            self.dynamic_array = self._shrink_stack()
        # type enforcement
        if not isinstance(value, self.datatype):
            raise TypeError(f"Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}")
        # add element to end of the stack
        self.dynamic_array[self.top] = value
        self.top += 1   # increment tracker

    def pop(self):
        """removes and returns the last item from the stack"""
        # is stack empty?
        if self.is_empty():
            raise IndexError("Stack is Empty!")
        deleted = self.dynamic_array[self.top-1]  # return deleted node
        if self.type is object or self.type is ctypes.py_object:
            self.dynamic_array[self.top-1] = None # dereference item
        self.top -= 1   # decrement array size.
        return deleted

    def peek(self):
        """returns (but not removes) the last item from the stack"""
        if self.is_empty():
            raise IndexError("Stack is Empty!")
        result = self.dynamic_array[self.top - 1]
        return result

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        """returns true if the stack is empty"""
        return self.top <= 0

    def __len__(self) -> int:
        """returns the amount of elements in the stack (not the capacity)"""
        return self.top

    def clear(self):
        """empties the current stack. (does not reset to the original state)"""
        while self.top > 0:
            self.pop()

    def __contains__(self, value):
        """Does the Stack contain this value? return boolean - overrides python builtin - Code: 25 in stack"""
        for i in range(self.top):
            if value == self.dynamic_array[i]:
                return True
        return False

    def __iter__(self):
        """Enables list like iteration with for loops etc over the stack elements. generator & uses yield"""
        for i in reversed(range(self.top)):
            yield self.dynamic_array[i]



# main --- client facing code ---
def main():
    # --- Edge Case Test Script for DynamicArrayStack ---

    # Dynamic test class
    DynamicClass = type("DynamicClass", (), 
    {
        "__init__": lambda self, name: setattr(self, "name", name),
        "__str__": lambda self: f"DynamicClass({self.name})"
    }
    )

    # --- Helper function to print stack info ---
    def print_stack_info(stack):
        print(stack)
        print(f"Length: {len(stack)}, Is empty? {stack.is_empty()}, Capacity: {stack.capacity}\n")

    # --- Test small initial capacities and auto-grow/shrink ---
    print("=== Testing int stack with small capacity (1) ===")
    int_stack_small = DynamicArrayStack[int](initial_capacity=1, datatype=int, datatype_map=CTYPES_DATATYPES)
    int_stack_small.push(1)
    print_stack_info(int_stack_small)
    int_stack_small.push(2)  # triggers grow
    print_stack_info(int_stack_small)
    int_stack_small.pop()  # triggers shrink?
    print_stack_info(int_stack_small)
    int_stack_small.pop()
    print_stack_info(int_stack_small)

    print("=== Testing string stack with initial capacity 2 ===")
    str_stack = DynamicArrayStack[str](initial_capacity=2, datatype=str, datatype_map=CTYPES_DATATYPES)
    str_stack.push("hello")
    str_stack.push("world")  # fill stack
    str_stack.push("!")      # triggers grow
    print_stack_info(str_stack)
    str_stack.pop()          # shrink possible?
    print_stack_info(str_stack)

    print("=== Testing list stack ===")
    list_stack = DynamicArrayStack[list](initial_capacity=2, datatype=list, datatype_map=CTYPES_DATATYPES)
    list_stack.push([1,2])
    list_stack.push([3,4])
    list_stack.push([5,6])  # grow
    print_stack_info(list_stack)
    list_stack.pop()         # shrink possible
    print_stack_info(list_stack)

    print("=== Testing dynamic class stack ===")
    dyn_stack = DynamicArrayStack[DynamicClass](initial_capacity=2, datatype=DynamicClass, datatype_map=CTYPES_DATATYPES)
    dyn_stack.push(DynamicClass("Alice"))
    dyn_stack.push(DynamicClass("Bob"))
    dyn_stack.push(DynamicClass("Charlie"))  # grow
    print_stack_info(dyn_stack)
    dyn_stack.pop()  # shrink possible
    print_stack_info(dyn_stack)

    print("=== Testing type enforcement ===")
    try:
        int_stack_small.push("string")  # should raise TypeError
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")

    try:
        str_stack.push(123)  # should raise TypeError
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")

    try:
        list_stack.push("not a list")  # should raise TypeError
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")

    try:
        dyn_stack.push(42)  # should raise TypeError
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")

    print("=== Testing iteration ===")
    for item in dyn_stack:
        print("Iterated item:", item)

    print("=== Testing clear ===")
    dyn_stack.clear()
    print_stack_info(dyn_stack)

if __name__ == "__main__":
    main()