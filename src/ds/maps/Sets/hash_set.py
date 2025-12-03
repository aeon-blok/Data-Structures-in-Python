# region standard lib
from types import UnionType
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
    Tuple,
    Literal,
    Iterable,
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
import uuid
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import (
    T,
    K,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    Index,
)
from user_defined_types.hashtable_types import (
    NormalizedFloat,
    LoadFactor,
    HashCodeType,
    CompressFuncType,
)
from user_defined_types.key_types import iKey, Key

from utils.constants import (
    MIN_HASHTABLE_CAPACITY,
    BUCKET_CAPACITY,
    HASHTABLE_RESIZE_FACTOR,
    DEFAULT_HASHTABLE_CAPACITY,
    MAX_LOAD_FACTOR,
)

from utils.validation_utils import DsValidation
from utils.representations import HashSetRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.set_adt import SetADT

from ds.maps.map_utils import MapUtils
from ds.maps.hash_functions import HashFuncConfig, HashFuncGen
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_open_addressing import HashTableOA

from user_defined_types.hashtable_types import SetSentinel
from user_defined_types.key_types import iKey, Key
from user_defined_types.generic_types import ValidDatatype, TypeSafeElement, Index, PositiveNumber


# endregion

"""
HashSet: A hash table implementation of a Set Data Structure
Elements are stored as keys in a hash table, with a dummy value (e.g., null or a sentinel object) associated with each key.
"""

class HashSet(SetADT[T], CollectionADT[T], Generic[T]):
    """
    Utilizes Composition for the underlying Hash Table Data structure.
    We use a Sentinel Value for the values() - so that they can be easily identified. (Stored in Hashable_types.py)
    """
    def __init__(self, datatype: type, capacity: int) -> None:
        self._datatype = ValidDatatype(datatype)
        self._set_capacity = PositiveNumber(capacity)   # just the initial capacity. dynamic after this.

        # composed objects:
        self._NIL = SetSentinel()
        self._ht = HashTableOA(SetSentinel, self._set_capacity)
        self._utils = MapUtils(self)
        self._validators = DsValidation()
        self._desc = HashSetRepr(self)

    @property
    def ht(self) -> HashTableOA:
        return self._ht

    @property
    def datatype(self) -> type:
        return self._datatype

    @property
    def return_elements(self) -> VectorArray:
        """returns the elements of the set as an array"""
        keys = self.ht.keys()
        total_keys = len(keys)
        elements = VectorArray(total_keys * 2, self._datatype)
        for i in keys:
            elements.append(i.value)
        return elements

    @property
    def return_key_objects(self) -> VectorArray:
        """returns key() objects which can be used to perform comparisons and sort by max, min etc.... they are hashable also."""
        return self.ht.keys()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return self._ht.total_elements

    def __contains__(self, element: T) -> bool:
        """
        Compute the hash of the element and check if the corresponding key exists in the hash table. 
        Average-case time complexity is O(1).
        """
        return self._ht.__contains__(element)

    def is_empty(self) -> bool:
        return self._ht.total_elements == 0

    def clear(self) -> None:
        self._ht.clear()

    def __iter__(self):
        return (i.value for i in self._ht.keys())

    # ----- Utility -----
    def __str__(self) -> str:
        return self._desc.str_hashset()

    def __repr__(self) -> str:
        return self._desc.repr_hashset()

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def __ior__(self, other: "HashSet[T]") -> "HashSet[T]":
        """In Place Union Operator A |= B """
        for item in other:
            self.add(item)  # add() already inserts into your hash table
        return self

    def __iand__(self, other: "HashSet[T]") -> "HashSet[T]":
        """In Place Intersection A &= B"""
        to_remove = [item for item in self if not other.__contains__(item)]
        for item in to_remove:
            self.remove(item)  # remove() removes from your hash table
        return self

    def __isub__(self, other: "HashSet[T]") -> "HashSet[T]":
        """In Place Difference: A -= B"""
        for item in other:
            if self.__contains__(item):
                self.remove(item)
        return self

    def __ixor__(self, other: "HashSet[T]") -> "HashSet[T]":
        """In Place Symm Difference: A ^= B"""
        for item in other:
            if self.__contains__(item):
                self.remove(item)
            else:
                self.add(item)
        return self

    def is_disjoint(self, other: SetADT[T]) -> bool:
        """compares two sets together - if they dont have any elements in common, they are said to be disjoint."""

        self._utils.validate_set(other)

        for element in self:
            if other.__contains__(element):
                return False

        for element in other:
            if self.__contains__(element):
                return False

        return True

    def subset(self, other: SetADT[T]) -> bool:
        """Checks if all elements in this set are in set B"""

        self._utils.validate_set(other)

        # * if the set is bigger than its comparison set, it cannot be a subset.
        if len(self) > len(other):
            return False

        # * check to see if all the items in this set are contained in comparison set.
        for element in self:
            if not other.__contains__(element):
                return False
        return True

    # ----- Mutators -----
    def add(self, element: T) -> None:
        """
        Compute the hash of the element and store it as a key. 
        If a key already exists, the element is not added (ensuring uniqueness). 
        Average-case time complexity is O(1).
        """
        # validate input
        element = TypeSafeElement(element, self._datatype)

        # * element already exists case: the get() method has a default value of None.
        search_result = self._ht.get(element) 
        if search_result is not None:
            return

        # * element doesnt exist case: add new element to the table.
        self._ht.put(element, self._NIL)

    def remove(self, element: T) -> None:
        """remove set element from set."""
        # validate input
        element = TypeSafeElement(element, self._datatype)
        # * remove element.
        self._ht.remove(element)

    def union(self, other: SetADT[T]) -> SetADT[T]:
        """Combines the elements of 2 unique sets, into a new set. O(n + m)"""

        self._utils.validate_set(other)

        # initialize new set
        new_capacity = (len(self) + len(other)) * 2
        new_set = HashSet(self._datatype, new_capacity)

        # add elements. - internal logic wi
        for element in self:
            new_set.add(element)
        for element in other:
            new_set.add(element)

        return new_set

    def intersection(self, other: SetADT[T]) -> SetADT[T]:
        """If an element exists in both sets, add to a new set."""

        self._utils.validate_set(other)

        new_capacity = (len(self) + len(other)) * 2
        new_set = HashSet(self._datatype, new_capacity)

        for element in self:
            if other.__contains__(element):
                new_set.add(element)

        return new_set

    def difference(self, other: SetADT[T]) -> SetADT[T]:
        """Elements that exist in set A, but not set B. add these to a new set."""

        self._utils.validate_set(other)

        new_capacity = (len(self) + len(other)) * 2
        new_set = HashSet(self._datatype, new_capacity)

        for element in self:
            if not other.__contains__(element):
                new_set.add(element)

        return new_set

    def symmetric_difference(self, other: SetADT[T]) -> SetADT[T]:
        """the elements that exist in set A or set B, but not in Both sets at the same time, add these to a new set and return it."""

        self._utils.validate_set(other)

        new_capacity = (len(self) + len(other)) * 2
        new_set = HashSet(self._datatype, new_capacity)

        for element in self:
            if other.__contains__(element):
                continue
            new_set.add(element) 

        for element in other:
            if self.__contains__(element):
                continue
            new_set.add(element)

        return new_set

    # region set operators
    # * ----- Set Operator Python Overrides -----
    __or__ = union  # A | B -- is a reference to the method, not a call. no brackets needed
    __and__ = intersection # A & B
    __sub__ = difference # A - B
    __xor__ = symmetric_difference # A ^ B
    # endregion


# Main --------------- Client Facing Code --------------------

def main():

    person_names_with_fruits = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Eve",
        "Frank",
        "Grace",
        "Hank",
        "Ivy",
        "Jack",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
    ]

    string_data = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli",
        "watermelon",
    ]

    set_a = HashSet(str, 10)
    print(set_a)
    print(repr(set_a))
    print(f"Is set empty? {set_a.is_empty()}")

    for i in string_data:
        set_a.add(i)

    print(set_a)
    print(repr(set_a))

    set_b = HashSet(str, 10)
    print(set_b)
    print(repr(set_b))

    for i in person_names_with_fruits:
        set_b.add(i)

    print(set_b)
    print(repr(set_b))

    print(f"\nTesting Union Operation and set operators: -- (merges both sets)")
    set_c = set_a | set_b
    print(type(set_c).__name__)
    print(set_c)
    print(repr(set_c))

    for i in string_data:
        set_c.remove(i)
    print(f"After removals....")
    print(set_c)
    print(f"Testing Contains: is Frank in Set? {'Frank' in set_c}")
    print(f"comparing sets to see if disjoint: {set_a.is_disjoint(set_b)}")

    print(f"\nTesting Intersection Operation and set operator -- (items contained in both sets)")
    set_d = set_a & set_b 
    print(set_d)
    print(repr(set_d))

    print(f"\nTesting Difference Operation: (items that exist in set a but not set b)")
    set_e = set_c - set_a
    print(set_e)
    print(repr(set_e))

    print(f"\nTesting Symmetric Difference Operation: (items that are not in both sets at the same time.)")
    set_f = set_b ^ set_e
    print(set_b)
    print(set_e)
    print(set_f)

    check_subset = set_e.subset(set_b)
    print(f"is Subset? {check_subset}")
    check_subset = set_a.subset(set_b)
    print(f"is Subset? {check_subset}")


    print(f"\nTesting return of key objects for comparisons....")
    set_a_keys = set_a.return_key_objects
    print(f"{set_a_keys}")

    print(f"\nTesting return of elements... as an array.")
    set_a_elements = set_a.return_elements
    print(f"{set_a_elements}")

    print(f"\nTesting Clear")
    set_a.clear()
    print(set_a)


if __name__ == "__main__":
    main()
