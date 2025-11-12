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


class MinMaxStack(StackADT[T], CollectionADT[T], Generic[T]):
    """A Min/Max Stack is a stack variant that can return the current minimum or maximum element (as defined by comparison operators) from the stack in O(1) time"""

    def __init__(self, datatype: type, key: Optional[Callable[[T],T]] = None, capacity: int = 10) -> None:
        self._datatype = datatype
        self._capacity = capacity
        self._stack = VectorArray(self._capacity, self._datatype)
        self._min_stack = VectorArray(self._capacity, self._datatype)
        self._max_stack = VectorArray(self._capacity, self._datatype)
        self.key = key
        self._top: int = -1  # starts at -1 -- not a valid index.
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
    @property
    def min(self):
        return self._min_stack.array[self._top]
    @property
    def max(self):
        return self._max_stack.array[self._top]

    # ------------ Utilities ------------
    def __str__(self) -> str:
        return self._desc.str_min_max_stack()

    def __repr__(self) -> str:
        return self._desc.repr_array_stack()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self.size

    def __contains__(self, value: T) -> bool:
        for i in range(self.size):
            if self._stack.array[i] == value:
                return True
        return False

    def is_empty(self) -> bool:
        return self.size == 0

    def clear(self) -> None:
        self._stack = VectorArray(self._capacity, self._datatype)
        self._min_stack = VectorArray(self._capacity, self._datatype)
        self._max_stack = VectorArray(self._capacity, self._datatype)
        self._top = -1

    def __iter__(self) -> Generator[T, None, None]:
        for i in range(self.size):
            yield self._stack.array[i]

    def __reversed__(self):
        for i in range(self.size - 1, -1, -1):
            yield self._stack.array[i]

    # ----- Canonical ADT Operations -----
    def push(self, element: T) -> None:
        """
        Insert an element at the top -- compares new element to previous top to determine new min and max, and adds them to min and max stacks.
        """
        self._validators.enforce_type(element, self.datatype)
        self._stack.append(element)
        # Empty Stack Case: if there are no elements - add the current element to the min max stacks - its the new min & max!
        if self._min_stack.is_empty():
            self._top += 1  # tracks the top of the stack.
            self._min_stack.append(element)
            self._max_stack.append(element)

        else:
            # Default Case: Calculate the min and max elements and append to stack.
            # set the key either using custom key (function) or preset defaults.
            if not self.key:
                self.key = self._utils.min_max_standard_comparison_lib()
            key_function = self.key

            # get current min and max
            current_min = self._min_stack.array[self._top]    
            current_max = self._max_stack.array[self._top]
            # set new min and max - by comparing the results of our key function with the current min and max.
            new_min_key = key_function(element)
            current_min_key = key_function(current_min)
            new_max_key = key_function(element)
            current_max_key = key_function(current_max)

            if not isinstance(new_min_key, type(current_min_key)):
                raise KeyInvalidError(f"Key is Invalid Type for Comparison.")
            if not isinstance(new_max_key, type(current_max_key)):
                raise KeyInvalidError(f"Key is Invalid Type for Comparison.")

            new_min = element if new_min_key < current_min_key else current_min
            new_max = element if new_max_key > current_max_key else current_max

            # append the new min and max to their respective stacks.
            self._top += 1  # tracks the top of the stack.
            self._min_stack.append(new_min)
            self._max_stack.append(new_max)

    def pop(self) -> T:
        """remove and return an element from the top"""
        self._utils.check_array_stack_underflow_error()
        old_value = self._stack.array[self._top]
        self._top -= 1
        if self._datatype in (object, ctypes.py_object):
            self._stack.array[self._top + 1] = None
            self._min_stack.array[self._top + 1] = None
            self._max_stack.array[self._top + 1] = None
        return old_value

    def peek(self) -> T:
        """return but dont remove an element from the top"""
        self._utils.check_array_stack_underflow_error()
        return self._stack.array[self._top]


# main ---- client facing code ----
def main():
    print("Integer Stack:")
    stack = MinMaxStack(int)
    for x in [3, 1, 4, 2]:
        stack.push(x)
        print(stack)

    while not stack.is_empty():
        stack.pop()
        print(stack)

    print("\nList Stack:")
    list_stack = MinMaxStack(list)
    list_stack.push([1, 2, 3])
    list_stack.push([4, 5])
    list_stack.push([6, 7, 8, 9])
    print(list_stack)

    print("\nTuple Stack (lexicographic comparison):")
    tuple_stack = MinMaxStack(tuple)
    for t in [(1, 2), (0, 5), (2, 1)]:
        tuple_stack.push(t)
    print(tuple_stack)

    # ----- Custom Class Stack -----
    class MyObj:
        def __init__(self, val):
            self.val = val
        def __repr__(self):
            return f"MyObj({self.val})"

    print("\nCustom Class Stack (key=lambda x: x.val):")
    obj_stack = MinMaxStack(MyObj, key=lambda x: x.val)
    for o in [MyObj(10), MyObj(3), MyObj(7)]:
        obj_stack.push(o)
    print(obj_stack)

    print("\nMulti Stack - Comparing iterables by number of elements")
    key = lambda x: len(x)
    multi_stack = MinMaxStack[Any](object, key=key)  # generic object stack
    multi_stack.push([1, 2, 3, 25, 50, 100])
    multi_stack.push({"a": 1, "b": 2, "c": 3, "d":4})
    multi_stack.push((2,3,4))
    multi_stack.push({1,90})

    print(multi_stack)

    # Mixed classes
    print("\nMixed Classes Stack: Comparing different classes and different attrributes")
    class Product:
        def __init__(self, cost):
            self.cost = cost
        def __repr__(self) -> str:
            return f"Product({self.cost})"

    class RawMaterial:
        def __init__(self, weight):
            self.weight = weight
        def __repr__(self) -> str:
            return f"RawMaterial({self.weight})"

    # key must normalize them
    key = lambda x: x.cost if isinstance(x, Product) else x.weight
    multi_class_stack = MinMaxStack(object, key=key)
    multi_class_stack.push(Product(100))
    multi_class_stack.push(RawMaterial(7))
    multi_class_stack.push(Product(50))
    multi_class_stack.push(Product(50))
    multi_class_stack.push(Product(23))
    multi_class_stack.push(Product(5012))
    multi_class_stack.push(Product(509))
    multi_class_stack.push(RawMaterial(75))
    multi_class_stack.push(RawMaterial(37))
    multi_class_stack.push(RawMaterial(98))
    print(multi_class_stack)

    # ----- Underflow Test -----
    print("\nUnderflow Test:")
    empty_stack = MinMaxStack(int)
    try:
        empty_stack.pop()
    except Exception as e:
        print(f"Caught expected error: {e}")
        
    print("\nStrings Test:")
    stack = MinMaxStack(str)
    stack.push("apple")
    stack.push("banana")
    stack.push("aardvark")
    stack.push("Welcome!")
    stack.push("Narayadana")
    stack.push("Obelisk")
    stack.push("Scifi")
    print(stack)  # Shows all elements


if __name__ == "__main__":
    main()
