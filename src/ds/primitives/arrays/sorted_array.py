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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, ValidDatatype, ValidIndex, Index, TypeSafeElement
from user_defined_types.array_types import BSearch
from user_defined_types.key_types import iKey, Key

from utils.validation_utils import DsValidation
from utils.representations import ArrayRepr, ViewRepr, SortedArrayRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.primitives.arrays.array_utils import ArrayUtils
from ds.algorithms.binary_search import BinarySearch

# endregion


"""
Sorted Array: invented by von neumann
Every element in the array is sorted by ascending or descending order.
This means that every element inside the array must be comparable to the other elements in the array.

Properties:
Sorted Arrays can contain duplicate elements. -- This makes searches for find first occurence and find last occurence more complex.
"""

class SortedArray(CollectionADT[T]):
    """
    Sorted Array Data Structure:
    Utilizes a Key() wrapper for objects as a way to enforce comparability for elements in the array.
    Via Inheritance - only insert needs to be overriden from a standard array in order to maintain the sorted invariant.
    """
    def __init__(self, datatype: type, capacity: int) -> None:
        self._datatype = datatype

        # composed objects
        self._utils = ArrayUtils(self)
        self._validators = DsValidation()
        self._desc = SortedArrayRepr(self)
        # we store keys in the array rather than the pure elements.
        self._array = VectorArray(capacity, iKey) 
        # we only search on the elements that are in the array. not the total capacity. (more efficient.)
        self._binary_search = BinarySearch()

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def size(self):
        """Total Elements in the array"""
        return self._array.size

    @property
    def array(self):
        """the underlying array"""
        return self._array.array

    @property
    def capacity(self):
        """Current Capacity of the array."""
        return self._array.capacity

    @property
    def is_static(self):
        return self._array.is_static

    @property
    def descending_order(self) -> VectorArray[T]:
        """returns an array of the elements in descending order."""
        descend = VectorArray(self.size, self.datatype)
        for i in range(self.size - 1, -1, -1):
            key = self._array.array[i]
            element = key.value
            descend.append(element)
        return descend

    @property
    def return_keys(self):
        """returns the key objects for easy comparison"""
        keys = VectorArray(self._array.size, iKey)
        for i in self:
            keys.append(i)
        return keys

    # ---------------- Meta Collection ADT Operations ----------------
    def __len__(self) -> Index:
        return self._array.size

    def is_empty(self) -> bool:
        return self._array.is_empty()

    def clear(self) -> None:
        self._array.clear()

    def __contains__(self, element: T) -> bool:
        key = Key(element)
        return self._array.__contains__(key)

    def __iter__(self):
        for i in range(self._array.size):
            key = self._array.array[i]
            element = key.value
            yield element

    def __reversed__(self):
        """reverses the iteration"""
        for i in range(self.size - 1, -1, -1):
            key = self._array.array[i]
            element = key.value
            yield element

    def __bool__(self):
        """allows for existence checks with while self: """
        return self.size > 0

    # ---------------- Utility Operations ----------------

    def __str__(self) -> str:
        return self._desc.str_array()

    def __repr__(self) -> str:
        return self._desc.repr_array()

    def __getitem__(self, index):
        return self.get(index)

    def __delitem__(self, index):
        return self.delete(index)

    # ---------------- Standard Operations ----------------

    # ---------------- Accessor Operations ----------------
    def get(self, index) -> T:
        """searches via index and returns the element"""
        key = self._array.get(index)
        element = key.value # type: ignore
        return element

    def min_value(self) -> T:
        """returns the minimum value in the array."""
        key = self._array.array[0]
        element = key.value
        return element

    def max_value(self) -> T:
        """returns the maximum value in the array."""
        array_length = self.size
        key = self._array.array[array_length-1]
        element = key.value
        return element

    def binary_search(self, target_value: T, search_type: BSearch = BSearch.INTERPOLATION) -> Optional[Index]:
        """
        Binary Search Algorithm implemented into sorted array via object composition
        allows a choice between several different binary search algos - like exponential, interpolated etc
        """
        target_value = TypeSafeElement(target_value, self.datatype)
        key = Key(target_value)

        if search_type == BSearch.CLASSIC:
            return self._binary_search.classic_binary_search(key, self._array.array[:self.size])
        elif search_type == BSearch.EXPONENTIAL:
            return self._binary_search.binary_exponential_search(key, self._array.array[:self.size])
        elif search_type == BSearch.INTERPOLATION:
            if not isinstance(key.value, (int, float)):
                raise DsTypeError("Interpolation search only works on numeric keys.")
            return self._binary_search.binary_interpolation_search(key, self._array.array[:self.size])
        elif search_type == BSearch.RECURSIVE:
            return self._binary_search.recursive_binary_search(key, self._array.array[:self.size])
        else:
            raise DsTypeError(f"Error: Search Type Must be valid Binary Search Type. Check Array Types in User Defined Types.")

    def lower_bounds(self, target_value: T) -> Optional[Index]:
        """return the first index where the index is greater than or EQUAL to the specified target value."""
        target_value = TypeSafeElement(target_value, self.datatype)
        key = Key(target_value)

        return self._binary_search.binary_search_lower_bounds(key, self._array.array[:self.size])

    def upper_bounds(self, target_value: T) -> Optional[Index]:
        """return the first index where the index is strictly greater than the target value. """
        target_value = TypeSafeElement(target_value, self.datatype)
        key = Key(target_value)

        return self._binary_search.binary_search_upper_bounds(key, self._array.array[:self.size])

    def rank_query(self, target_value: T) -> Optional[Index]:
        """the number of elements that are less than the specified target value -- O(logN)"""
        target_value = TypeSafeElement(target_value, self.datatype)
        key = Key(target_value)

        return self._binary_search.binary_search_rank(key, self._array.array[:self.size])

    def predecessor(self, element: T) -> Optional[Index]:
        """
        returns the specified element's predecessor INDEX VALUE -- the largest element that is smaller than the specified element.

        """
        target_value = TypeSafeElement(element, self.datatype)
        key = Key(target_value)

        return self._binary_search.binary_search_predecessor(key, self._array.array[:self.size])

    def successor(self, element: T) -> Optional[Index]:
        """
        returns the specified elements successor INDEX VALUE-- the smallest element that is larger than the specified element
        """
        target_value = TypeSafeElement(element, self.datatype)
        key = Key(target_value)

        return self._binary_search.binary_search_successor(key, self._array.array[:self.size])

    # ---------------- Mutator Operations ----------------
    def insert(self, element: T) -> None:
        """
        Find the correct index using binary search (lower bound).
        then insert as usual - composed object array will handle the details...
        """
        element = TypeSafeElement(element, self.datatype)
        # pack item into key object - this validates that elements are comparable.
        key = Key(element)

        # * empty array case:
        if self.size == 0:
            self._array.insert(0, key)
        # * main case
        else:
            derived_index = self._binary_search.binary_search_lower_bounds(key, self._array.array[:self.size])
            self._array.insert(derived_index, key)

    def delete(self, index: Index) -> T:
        """Delete element -- handled by underlying composed array object."""
        deleted_key = self._array.delete(index)
        deleted_element = deleted_key.value
        return deleted_element


# ------------------------------ Main: Client Facing Code: ------------------------------

# todo - add bulk insert functionality to take advantages of fast Binary Search
# todo - maybe add setitem - see how to do it, or raise exception if someone tries to use it....

def main():
    # Create a sorted array of integers with initial capacity 10
    sa = SortedArray(int, 10)

    # Insert elements in random order
    elements = [5, 1, 3, 7, 2, 6, 4, 25, 905, 3343, 384, 543,58695046, 543, 22, 411, 198, 70777, 583, 393485, 322, 112]
    print("Inserting elements:", elements)
    for i in elements:
        sa.insert(i)

    # Print the array after insertion
    print("SortedArray contents after insertions:", sa)
    print(repr(sa))
    print(f"Descending order copy.", sa.descending_order)

    # Test min and max
    print("Min value:", sa.min_value())
    print("Max value:", sa.max_value())

    # Test binary search (classic)
    print(f"Testing Binary Search Algorithm: ")
    for target in [1, 4, 6, 8]:
        index = sa.binary_search(target, search_type=BSearch.EXPONENTIAL)
        index_val = sa[index] if index is not None else None
        print(f"Binary search for {target}: index = {index} Verify: {index_val}")

    # Test lower_bounds and upper_bounds
    target = 4
    print(f"Lower bound for {target}:", sa.lower_bounds(target))
    print(f"Upper bound for {target}:", sa.upper_bounds(target))

    # Test rank query
    target = 5
    print(f"Rank query for {target} (number of elements < {target}):", sa.rank_query(target))

    # Test predecessor and successor
    print(f"Testing Predecessor and Successor...")
    print(sa)
    for target in [1, 4, 7, 8]:
        pred = sa.predecessor(target)
        succ = sa.successor(target)
        pred_val = sa[pred] if pred is not None else None
        succ_val = sa[succ] if succ is not None else None
        print(f"Element: {target}, Predecessor index: {pred}, Verify: {pred_val}, Successor index: {succ}, verify: {succ_val}")

    # Test deletion
    print("Deleting element at index 3:", sa.delete(3))
    print("Array after deletion:", sa)

    # Test contains
    print("Contains 5?", 5 in sa)
    print("Contains 10?", 10 in sa)

    # Test clear
    sa.clear()
    print("Array after clear:", list(sa))

    # ? ------------------------- Create a sorted array of strings -------------------------
    print(f"\nTesting Strings with Sorted Array")
    sa_str = SortedArray(str, 10)

    # Insert string elements in random order
    elements = ["fig", "banana", "date", "cherry", "grape", "apple"]
    print("Inserting string elements:", elements)
    for e in elements:
        sa_str.insert(e)

    # Print the array after insertion
    print("SortedArray contents after insertions:", sa_str)
    print(repr(sa_str))
    print(f"Descending order copy", sa_str.descending_order)

    # Test min and max
    print("Min value:", sa_str.min_value())
    print("Max value:", sa_str.max_value())

    # Test binary search (classic)
    print("\nBinary Search (Classic and Exponential) on strings:")
    for target in ["apple", "cherry", "kiwi"]:
        for search_type in [BSearch.CLASSIC, BSearch.EXPONENTIAL]:
            index = sa_str.binary_search(target, search_type=search_type)
            found_val = sa_str[index] if index is not None else None
            print(f"{search_type.name} search for '{target}': index={index}, found={found_val}")

    # Test lower and upper bounds
    target = "cherry"
    print(f"\nLower bound for '{target}':", sa_str.lower_bounds(target))
    print(f"Upper bound for '{target}':", sa_str.upper_bounds(target))

    # Test predecessor and successor
    print("\nPredecessor and Successor:")
    for target in ["apple", "banana", "cherry", "date", "kiwi"]:
        pred = sa_str.predecessor(target)
        succ = sa_str.successor(target)
        pred_val = sa_str[pred] if pred is not None else None
        succ_val = sa_str[succ] if succ is not None else None
        print(f"Element: '{target}', Predecessor index: {pred}, value: {pred_val}, "
              f"Successor index: {succ}, value: {succ_val}")


if __name__ == "__main__":
    main()
