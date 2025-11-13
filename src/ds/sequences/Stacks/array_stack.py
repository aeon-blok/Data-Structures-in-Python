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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from utils.constants import CTYPES_DATATYPES, NUMPY_DATATYPES, SHRINK_CAPACITY_RATIO
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import ArrayStackRepr
from utils.exceptions import *
from utils.helpers import RandomClass

from adts.collection_adt import CollectionADT
from adts.sequence_adt import SequenceADT
from adts.stack_adt import StackADT

from ds.primitives.arrays.array_utils import ArrayUtils
from ds.primitives.arrays.dynamic_array import VectorArray
from ds.sequences.Stacks.stack_utils import StackUtils

# endregion


class ArrayStack(StackADT[T], CollectionADT[T], Generic[T]):
    """Dynamically sized array based stack. The array will resize itself when it gets close to full."""
    def __init__(self, datatype:type, capacity: int = 10) -> None:
        self._datatype = datatype
        self._capacity = capacity
        self._array = VectorArray(self._capacity, self._datatype)
        self._top: int = -1 # starts at -1 -- not a valid index.
        # Composed Objects
        self._utils = StackUtils(self)
        self._validators = DsValidation()
        self._desc = ArrayStackRepr(self)

    @property
    def top(self):
        return self._top
    @property
    def datatype(self):
        return self._datatype
    @property
    def size(self):
        return self._top + 1

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_array_stack()

    def __repr__(self) -> str:
        return self._desc.repr_array_stack()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.size

    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            if self._array.array[i] == value:
                return True
        return False

    def is_empty(self) -> bool:
        return self.size == 0

    def clear(self) -> None:
        self._array = VectorArray(self._capacity, self._datatype)
        self._top = -1

    def __iter__(self) -> Generator[T, None, None]:
        for i in range(self.size):
            yield self._array.array[i]

    def __reversed__(self):
        for i in range(self.size-1, -1, -1):
            yield self._array.array[i]

    # ----- Canonical ADT Operations -----
    def push(self, element: T) -> None:
        """Insert an element at the top"""
        self._validators.enforce_type(element, self.datatype)
        self._top += 1  # tracks the top of the stack.
        self._array.append(element)

    def pop(self) -> T:
        """remove and return an element from the top"""
        self._utils.check_array_stack_underflow_error()
        old_value = self._array.array[self._top]
        self._top -= 1

        # dereference if object
        if self._datatype in (object, ctypes.py_object):
            self._array.array[self._top + 1] = None

        return old_value

    def peek(self) -> T:
        """return but dont remove an element from the top"""
        self._utils.check_array_stack_underflow_error()
        return self._array.array[self._top]


# todo test with class objects - for dereferencing.


# main --- client facing code ---
def main():
    print("=== DynamicArrayStack Test Suite ===\n")
    stack = ArrayStack(int)
    print(stack)
    print(f"Testing is_empty? {stack.is_empty()}\n")

    # Push Operations
    print("Testing Push Operations:")
    for val in [10, 20, 30, 40]:
        print(f"Pushing {val}...")
        stack.push(val)
        print(stack)

    # Peek Top Element
    try:
        print(f"\nPeek Top Element: {stack.peek()}")
        print(f"Current Top Property: {stack.top}")
    except Exception as e:
        print(f"Error: {e}")

    # Pop Operations
    print("\nTesting Pop Operations:")
    try:
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
    except Exception as e:
        print(f"Error: {e}")

    # Mixed Push/Pop
    print("\nTesting Mixed Push/Pop:")
    stack.push(50)
    print(stack)
    try:
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
    except Exception as e:
        print(f"Error: {e}")

    # Iteration
    print("\nTesting Iteration:")
    for val in [60, 70, 80]:
        stack.push(val)

    print("Iterating over stack (bottom -> top):")
    for i, item in enumerate(stack):
        print(f"Iterated over {i}: {item}")

    # Reversed Iteration
    print("\nIterating over stack (top -> bottom):")
    for i, item in enumerate(reversed(stack)):
        print(f"Iterated over {i}: {item}")

    # Contains
    print("\nTesting __contains__:")
    print(f"Is 70 in stack? {70 in stack}")
    print(f"Is 999 in stack? {999 in stack}")

    # Length
    print("\nTesting __len__:")
    print(f"Stack length: {len(stack)}")

    try:
        print("Pushing a wrong type into stack...")
        stack.push(RandomClass("Woaggggg"))  # should raise TypeError
    except TypeError as e:
        print(f"{e}")

    # Clear
    print("\nTesting clear():")
    stack.clear()
    print(stack)
    print(f"Testing is_empty? {stack.is_empty()}")

    # Pop/Peek from empty stack
    print("\nTesting errors on empty stack:")

    try:
        stack.pop()
    except Exception as e:
        print(f"{e}")

    try:
        stack.peek()
    except Exception as e:
        print(f"{e}")

    print("\n=== DynamicArrayStack Test Complete ===")


if __name__ == "__main__":
    main()
