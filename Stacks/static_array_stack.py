from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator, Generator
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes

"""
Stack ADT (Array-Based) Specification:
A Stack is a collection of objects that are inserted & removed according to the LIFO principle
The end user will only ever interact with the top element of the Stack.

Operations:
push(x) — Insert x on top.
pop() — Remove and return top element.
peek() — Return top element without removing.
is_empty() — Return True if stack is empty.
size() — Return number of elements in stack.

Invariants:
LIFO order maintained.
Non-commutative insertion: push(push(S, x), y) != push(push(S, y), x)
Pop from empty stack raises error.
Overflow raises OverflowError
"""


T = TypeVar('T')

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

class StaticArrayStack(StackADT[T]):
    """A Stack is a collection of elements that are accessed via LIFO principle. A Static Array is the underlying data structure for this stack - uses Ctype array."""

    def __init__(self, stack_capacity, datatype: type, datatype_map: dict = CTYPES_DATATYPES) -> None:

        if stack_capacity <=0:
            raise ValueError("Capacity must be greater than 0")

        # define the size of the stack (not the number of elements inside, but its capacity)
        self.stack_capacity = stack_capacity
        self.datatype_map = datatype_map
        self.datatype = datatype
        self.type = None
        self.static_array = None

        # chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)
        if datatype_map == CTYPES_DATATYPES:
            self.static_array = self._init_ctypes_array()
        elif datatype_map == NUMPY_DATATYPES:
            self.static_array = self._init_numpy_array()
        else:
            raise ValueError(f"Datatype Map Unknown... Map: {datatype_map}")

        # tracks the next empty slot in the stack.
        self.top = 0

    def _init_ctypes_array(self):
        """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
        # setting ctypes datatype -- needed for the array.
        if self.datatype not in self.datatype_map:
            self.type = ctypes.py_object
        else:
            self.type = CTYPES_DATATYPES[self.datatype] # maps type of array to ctype
        # creates a class object - an array of specified number of a specified type
        static_array_cls = self.type * self.stack_capacity
        self.static_array = static_array_cls() # initializes array with preallocated memory block
        return self.static_array

    def _init_numpy_array(self):
        """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
        if self.datatype not in NUMPY_DATATYPES:
            self.type = object
            self.static_array = numpy.empty(self.stack_capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            self.static_array = numpy.empty(self.stack_capacity, dtype=self.type)
        return self.static_array


    # ----- Utility -----
    def __str__(self) -> str:
        """a list of strings representing all the elements in the stack. Goes from Top to Bottom"""
        items = ', '.join(str(self.static_array[i]) for i in range(self.top-1, -1,-1))
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        return f"Type: {string_datatype}: Capacity: {self.top}/{self.stack_capacity} Stack Items: [Top]->> {items}"

    def capacity(self) -> int:
        """returns the total size(capacity) of the array. Not the number of elements that exist."""
        return self.stack_capacity

    def _overflow_error(self):
        """check stack capacity not exceeded"""        
        if self.top >= self.stack_capacity:
            raise OverflowError("Stack Capacity Exceeded...")

    def _underflow_error(self):
        """Check if stack is empty"""
        if self.is_empty():
            raise IndexError("Stack is Empty")

    def is_full(self):
        """returns true if stack is full to capacity."""
        return self.top >= self.stack_capacity
            

    # ----- Canonical ADT Operations -----
    def push(self, value):
        """Insert x on top."""
        self._overflow_error()
        # static type enforcement:
        if not isinstance(value, self.datatype):
            raise TypeError(f"Invalid Type, Expected: {self.datatype.__name__} Got: {type(value).__name__}")
    
        # assign new value the last index of the array
        self.static_array[self.top] = value
        self.top += 1  # increment stack size

    def pop(self):
        """Delete and return top element."""
        self._underflow_error()
        self.top -= 1   # decrement stack
        deleted_value = self.static_array[self.top]

        # if object or string need to dereference item
        if self.type == ctypes.py_object:  
            self.static_array[self.top] = None  

        return deleted_value

    def peek(self):
        """Return top element without removing."""
        return self.static_array[self.top - 1]


    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        """Return True if stack is empty."""
        return self.top == 0

    def __len__(self):
        """Return number of elements in stack."""
        return self.top

    def clear(self):
        """resets stack to original empty state"""
        while not self.is_empty():
            self.pop()

    def __contains__(self, value):
        """Does the stack contain this item? returns boolean"""
        for i in range(self.top):
            if self.static_array[i] == value:
                return True
        return False

    def __iter__(self):
        """Overrides iterator built in to iterate over stack. iterates from the top to bottom."""
        for i in reversed(range(self.top)):
            yield self.static_array[i]


# Main --- Client Facing Code ---
def main():

    def comprehensive_stack_tests():
        print("=== CTYPES Integer Stack ===")
        int_stack = StaticArrayStack[int](3, int)
        print("Empty stack:", int_stack)
        print("Is empty:", int_stack.is_empty())

        int_stack.push(10)
        int_stack.push(20)
        int_stack.push(30)
        print(int_stack)
        print("Is full:", int_stack.is_full())

        try:
            int_stack.push(40)
        except OverflowError as e:
            print("Overflow caught:", e)

        print("Peek:", int_stack.peek())
        print("Pop:", int_stack.pop())
        print("Contains 20:", 20 in int_stack)
        print("Iterating stack:", list(int_stack))
        int_stack.clear()
        print("After clear:", int_stack)
        print("Is empty after clear:", int_stack.is_empty())

        print("\n=== CTYPES String Stack ===")
        str_stack = StaticArrayStack[str](2, str)
        str_stack.push("hello")
        str_stack.push("world")
        print(str_stack)
        try:
            str_stack.push("overflow")
        except OverflowError as e:
            print("Overflow caught:", e)
        print("Pop:", str_stack.pop())
        print("Pop:", str_stack.pop())
        try:
            str_stack.pop()
        except IndexError as e:
            print("Underflow caught:", e)

        print("\n=== NUMPY Integer Stack ===")
        np_int_stack = StaticArrayStack[int](3, int, datatype_map=NUMPY_DATATYPES)
        np_int_stack.push(1)
        np_int_stack.push(2)
        np_int_stack.push(3)
        print(np_int_stack)
        print("Pop:", np_int_stack.pop())
        print(np_int_stack)

        print("\n=== NUMPY Float Stack ===")
        np_float_stack = StaticArrayStack[float](2, float, datatype_map=NUMPY_DATATYPES)
        np_float_stack.push(1.5)
        np_float_stack.push(2.5)
        print(np_float_stack)
        try:
            np_float_stack.push(3.5)
        except OverflowError as e:
            print("Overflow caught:", e)
        print("Peek:", np_float_stack.peek())
        print("Pop:", np_float_stack.pop())
        print("Pop:", np_float_stack.pop())
        print("Is empty:", np_float_stack.is_empty())

        print("\n=== NUMPY Object Stack ===")
        # Dynamically create a class
        Obama = type(
            "Obama",
            (object,),
            {
                "__init__": lambda self, name: setattr(self, "name", name),
                "__repr__": lambda self: f"MyClass(name={self.name})"
            }
            )
        np_obj_stack = StaticArrayStack[Obama](3, Obama, datatype_map=NUMPY_DATATYPES)

        # Valid instances
        a = Obama("a")
        b = Obama("b")
        c = Obama("c")

        np_obj_stack.push(a)
        np_obj_stack.push(b)
        np_obj_stack.push(c)

        # Overflow test
        try:
            np_obj_stack.push(Obama("d"))
        except OverflowError as e:
            print("Overflow caught:", e)


        # Type enforcement test
        print("Attempting to push invalid type (int)...")
        np_obj_stack.pop()
        try:
            np_obj_stack.push(42)
        except TypeError as e:
            print("TypeError caught:", e)
        print("Attempting to push invalid type (str)...")

        try:
            np_obj_stack.push("hello")
        except TypeError as e:
            print("TypeError caught:", e)

        # LIFO behavior
        print("Popping items (should be c, b, a)...")
        while not np_obj_stack.is_empty():
            print("Popped:", np_obj_stack.pop())

        # Underflow test
        try:
            np_obj_stack.pop()
        except IndexError as e:
            print("Underflow caught:", e)

        # Re-push after clear
        print("Re-pushing a valid instance after clearing...")
        np_obj_stack.push(Obama("x"))
        print(np_obj_stack)

        print("\n=== Dynamic Class Object Stack ===")
        MyClass = type(
            "MyClass",
            (object,),
            {
                "__init__": lambda self, name: setattr(self, "name", name),
                "__repr__": lambda self: f"MyClass(name={self.name})"
            }
        )

        obj_stack = StaticArrayStack[MyClass](3, MyClass)
        instances = [MyClass("a"), MyClass("b"), MyClass("c")]
        for obj in instances:
            obj_stack.push(obj)
        print(obj_stack)
        print("Iterating stack:", list(obj_stack))
        print("Pop:", obj_stack.pop())
        print("After pop:", obj_stack)
        obj_stack.clear()
        print("After clear:", obj_stack)

    # Run the comprehensive tests
    comprehensive_stack_tests()


if __name__ == "__main__":
    main()
