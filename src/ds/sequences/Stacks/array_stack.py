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
from user_defined_types.generic_types import T, Index
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
        self._data = VectorArray(self._capacity, self._datatype)
        self._top: int = -1 # starts at -1 -- not a valid index.
        # Composed Objects
        self._utils = StackUtils(self)
        self._validators = DsValidation()
        self._desc = ArrayStackRepr(self)

    @property
    def top(self) -> Index:
        return self._top
    @property
    def datatype(self):
        return self._datatype
    @property
    def size(self) -> int:
        return self._top + 1
    @property
    def data(self) -> VectorArray:
        return self._data

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_array_stack()

    def __repr__(self) -> str:
        return self._desc.repr_array_stack()

    def __bool__(self):
        return self.size > 0

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.size

    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            if self._data.array[i] == value:
                return True
        return False

    def is_empty(self) -> bool:
        return self.size == 0

    def clear(self) -> None:
        self._data = VectorArray(self._capacity, self._datatype)
        self._top = -1

    def __iter__(self) -> Generator[T, None, None]:
        for i in range(self.size):
            yield self._data.array[i]

    def __reversed__(self):
        for i in range(self.size-1, -1, -1):
            yield self._data.array[i]

    # ----- Canonical ADT Operations -----
    def push(self, element: T) -> None:
        """Insert an element at the top"""
        self._validators.enforce_type(element, self.datatype)
        self._data.append(element)
        self._top += 1  # tracks the top of the stack.

    def pop(self) -> T:
        """remove and return an element from the top"""
        self._utils.check_array_stack_underflow_error()
        old_value = self._data.array[self._top]
        self._data.delete(self._top)
        self._top -= 1
        return old_value

    def peek(self) -> T:
        """return but dont remove an element from the top"""
        self._utils.check_array_stack_underflow_error()
        return self._data.array[self._top]


# todo write more tests, including class objects. test errors also.


# main --- client facing code ---
def main():
    print("=== DynamicArrayStack Test Suite ===\n")
    stack = ArrayStack(int)
    print(stack)
    print(repr(stack))
    print(f"Testing is_empty? {stack.is_empty()}\n")
    for i in range(50):
        stack.push(i)
    print(stack)

    for i in range(40):
        stack.pop()
    print(stack)
    print(repr(stack))


if __name__ == "__main__":
    main()
