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
from faker import Faker

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
from utils.representations import OrderedRBSetRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.set_adt import SetADT
from adts.ordered_set_adt import OrderedSetADT

from ds.maps.map_utils import MapUtils
from ds.trees.tree_utils import TreeUtils
from ds.maps.hash_functions import HashFuncConfig, HashFuncGen
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_open_addressing import HashTableOA

from user_defined_types.hashtable_types import SetSentinel
from user_defined_types.key_types import iKey, Key
from user_defined_types.generic_types import (
    ValidDatatype,
    TypeSafeElement,
    Index,
    PositiveNumber,
)

# endregion

"""
Ordered Set: Implements an Ordered Set via a red black tree, 
has all the features of a set, plus ordered set adt interface
"""

class OrderedSet(OrderedSetADT[T], CollectionADT[T], Generic[T]):
    """
    Ordered Set Implementation using red black tree for its foundation.
    Elements are NOT stored in order, but they are returned in order - via inorder traversal.
    """
    def __init__(self, datatype: type) -> None:
        self._datatype = ValidDatatype(datatype)
        self._tree_keytype: type | None = None

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = OrderedRBSetRepr(self)
        # inline import to avoid circular import failure
        from ds.trees.Binary_Search_Trees.red_black_tree import RedBlackTree
        self._tree = RedBlackTree(self._datatype)

    @property
    def tree_keytype(self) -> Optional[type]:
        return self._tree_keytype

    @property
    def datatype(self) -> type:
        return self._datatype

    # ----- Meta Collection ADT Operations -----
    def clear(self) -> None:
        self._tree.clear()

    def __contains__(self, value: T) -> bool:
        element = TypeSafeElement(value, self._datatype)
        key = Key(element)
        return key in self._tree

    def __len__(self) -> Index:
        return len(self._tree)

    def is_empty(self) -> bool:
        return self._tree.is_empty()

    def __iter__(self):
        nodes = list(self._tree.inorder())
        for node in nodes:
            yield node.element

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_ordered_set()

    def __repr__(self) -> str:
        return self._desc.repr_ordered_set()

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def subset(self, other: SetADT[T]) -> bool:
        """checks whether the elements of this ordered set all belong to another set. returns a boolean"""
        for i in self:
            if i not in other:
                return False
        return True

    def min(self) -> T:
        """finds the smallest element (via comparator) in the ordered set."""

        # existence check
        if self._tree.is_empty():
            raise DsUnderflowError(f"Error: Ordered Set is Empty.")

        result = self._tree.minimum(self._tree.root)
        return result.element

    def max(self) -> T:
        """finds the largest element in the ordered set."""
        # existence check
        if self._tree.is_empty():
            raise DsUnderflowError(f"Error: Ordered Set is Empty.")

        result = self._tree.maximum(self._tree.root)
        return result.element

    def predecessor(self, element: T) -> T | None:
        """finds the largest element smaller than the specified element"""
        element = TypeSafeElement(element, self._datatype)
        key = Key(element)
        self._utils.check_key_is_same_type(key)

        node = self._tree.search(key)
        if node is None:
            return

        pred = self._tree.predecessor(node)
        if pred is self._tree.NIL:
            return None

        return pred.element

    def successor(self, element: T) -> T | None:
        """finds the smallest element larger than the specified element"""
        element = TypeSafeElement(element, self._datatype)
        key = Key(element)
        self._utils.check_key_is_same_type(key)

        node = self._tree.search(key)
        if node is None:
            return

        succ = self._tree.successor(node)
        if succ is self._tree.NIL:
            return None

        return succ.element

    def select_range(self, element_a: T, element_b: T) -> VectorArray[T] | None:
        """returns the elements of the ordered set that exist between the specified input parameters"""

        # existence check
        if self._tree.is_empty():
            return VectorArray(0, self._datatype)

        # validate inputs
        a = TypeSafeElement(element_a, self._datatype)
        b = TypeSafeElement(element_b, self._datatype)

        # validate keys
        key_a = Key(a)
        key_b = Key(b)
        self._utils.check_key_is_same_type(key_a)
        self._utils.check_key_is_same_type(key_b)

        # validate range boundaries
        if key_a > key_b: return
        max_node = self._tree.maximum(self._tree.root)
        # if input parameter is bigger than max value in ordered set - range is out of bounds
        if key_a > max_node.key: return

        # container
        result = VectorArray(len(self._tree), self._datatype)

        # * search for the smallest key greater than specified key.
        node = self._tree.find_lower_bounds(key_a)
        # existence check (for node)
        if node is self._tree.NIL:
            return VectorArray(0, self._datatype)

        # * traverse tree until we hit the upper boundary of range.
        while node is not self._tree.NIL and node.key <= key_b:
            result.append(node.element)
            node = self._tree.successor(node)

        return result

    # ----- Mutators -----
    def add(self, element: T) -> None:
        """
        Adds an element to the ordered set.
        returns none if the element already exists in the ordered set
        """

        # strong typing
        element = TypeSafeElement(element, self._datatype)
        key = Key(element)

        # validate keytype
        self._utils.set_keytype(key)
        self._utils.check_key_is_same_type(key)

        # existence check (rb tree uses key lookup)
        if key in self._tree:
            return

        # add to tree.
        self._tree.insert(key, element)

    def remove(self, element: T) -> None:
        """
        removes an element from the ordered set, 
        returns none if the element is not found
        """

        # strong typing
        element = TypeSafeElement(element, self._datatype)
        key = Key(element)

        # validate keytype
        self._utils.check_key_is_same_type(key)

        # existence check
        if key not in self._tree:
            return

        # delete from rb tree. (first have to search to get the node.)
        node = self._tree.search(key)
        old_element = self._tree.delete(node)

    def union(self, other: SetADT[T]) -> SetADT[T]:
        """
        returns a new set containing all elements in this ordered set and the specified input set
        the elements of this new set will be returned in order so must be hashable and comparable
        the elements for both sets must be the same datatype.
        """

        if not isinstance(other, SetADT):
            raise DsTypeError(f"Error: Input must be a Set Type and implement the SetADT interface")

        # type check
        if self._datatype is not other.datatype:
            raise DsTypeError(f"Error: Both Sets must have the same datatype. Expected {self._datatype}, Got: {other.datatype}")

        # container
        result = OrderedSet(self._datatype)

        for i in self:
            result.add(i)

        for i in other:
            result.add(i)

        return result

    def intersection(self, other: SetADT[T]) -> SetADT[T]:
        """
        returns a new set containing elements that exist in both sets.
        the original sets are not modified
        """

        if not isinstance(other, SetADT):
            raise DsTypeError(f"Error: Input must be a Set Type and implement the SetADT interface")

        # type check
        if self._datatype is not other.datatype:
            raise DsTypeError(f"Error: Both Sets must have the same datatype. Expected {self._datatype}, Got: {other.datatype}")

        result = OrderedSet(self._datatype)

        # iterate over the smaller set of the two sets.
        if len(self) <= len(other):
            for i in self:
                if i in other:
                    result.add(i)
        else:
            for i in other:
                if i in self:
                    result.add(i)

        return result

    def difference(self, other: SetADT[T]) -> SetADT[T]:
        """returns a new set containing elements that exist in the ordered set, but not in the specified input set."""

        if not isinstance(other, SetADT):
            raise DsTypeError(f"Error: Input must be a Set Type and implement the SetADT interface")

        # type check
        if self._datatype is not other.datatype:
            raise DsTypeError(f"Error: Both Sets must have the same datatype. Expected {self._datatype}, Got: {other.datatype}")

        result = OrderedSet(self._datatype)

        for i in self:
            if i not in other:
                result.add(i)

        return result

    def symmetric_difference(self, other: SetADT[T]) -> SetADT[T]:
        """returns a new set containing elements that exist in 1 set or the other, but not both."""

        if not isinstance(other, SetADT):
            raise DsTypeError(f"Error: Input must be a Set Type and implement the SetADT interface")

        # type check
        if self._datatype is not other.datatype:
            raise DsTypeError(f"Error: Both Sets must have the same datatype. Expected {self._datatype}, Got: {other.datatype}")

        result = OrderedSet(self._datatype)

        # A - B
        for i in self:
            if i not in other:
                result.add(i)

        # B - A
        for i in other:
            if i not in self:
                result.add(i)

        return result

    def split(self, seperator_element: T) -> Tuple:
        """
        Splits the ordered set into two seperate ordered sets, the specified element operates as the seperator value for these sets. 
        returns a tuple (ordered_set_a, ordered_set_b) 
        if any element matches the specified seperator element - it will be discarded
        """

        # type check
        seperator = TypeSafeElement(seperator_element, self._datatype)
        key = Key(seperator)
        self._utils.check_key_is_same_type(key)

        left = OrderedSet(self._datatype)
        right = OrderedSet(self._datatype)

        # equal elements to the seperatorelements are discarded
        for i in self:
            if i < seperator:
                left.add(i)
            elif i > seperator:
                right.add(i)

        return (left, right)

    def join(self, other: OrderedSetADT[T]):
        """
        Merges 2 sets into a single ordered set, 
        with the condition that the maximum element of set 1 is smaller than the minimum element of set 2
        """

        if not isinstance(other, OrderedSet):
            raise DsTypeError(f"Error: Input must be a Set Type and implement the SetADT interface")

        # type check
        if self._datatype is not other.datatype:
            raise DsTypeError(f"Error: Both Sets must have the same datatype. Expected {self._datatype}, Got: {other.datatype}")

        # existence check:
        if self.is_empty():
            return other

        if other.is_empty():
            return self

        # * Order Constraint
        set_a_max = self.max()
        set_b_min = other.min()

        if set_a_max >= set_b_min:
            raise DsInputValueError(f"Error: Merging ordered sets requires set 1 max element to be smaller than the min element of set 2.")

        result = OrderedSet(self._datatype)
        # add elements to new set.
        for i in self:
            result.add(i)
        for i in other:
            result.add(i)
        return result

    # region set operators
    # * ----- Set Operator Python Overrides -----
    __or__ = union  # A | B -- is a reference to the method, not a call. no brackets needed
    __and__ = intersection # A & B
    __sub__ = difference # A - B
    __xor__ = symmetric_difference # A ^ B
    # endregion


# Main --------------- Client Facing Code --------------------
def main():
    fake = Faker()
    fake.seed_instance(92)
    data = []
    for i in range(20):
        data.append(fake.word())
    print(f"Data: {data}")

    data_b = []
    for i in range(5):
        data_b.append(fake.word())
        data_b.append(data[i])
    print(f"Data_B: {data_b}")

    print(f"Creating Ordered set")
    ordered_set = OrderedSet(str)
    print(ordered_set)
    print(repr(ordered_set))
    print(f"Is ordered set empty? {ordered_set.is_empty()}")
    print(f"does ordered set contain this item? {'ahfdgkfdgdf' in ordered_set}")

    print(f"Testing Insertion into ordered set")
    for i in data:
        ordered_set.add(i)
    print(ordered_set)
    print(repr(ordered_set))

    print(f"Testing Deletion of items in ordered set")
    for i in ordered_set:
        ordered_set.remove(i)
    print(ordered_set)
    print(repr(ordered_set))

    print(f"adding elements back to set")
    for i in data:
        ordered_set.add(i)
    print(ordered_set)
    print(repr(ordered_set))

    random_elem_a = random.choice(list(ordered_set))
    random_elem_b = random.choice(list(ordered_set))
    print(f"does ordered set contain this item? {random_elem_b}={random_elem_b in ordered_set}")

    print(f"Testing Max Element: {ordered_set.max()}")
    print(f"Testing Min Element: {ordered_set.min()}")
    print(f"Testing Predecessor of [{random_elem_a}]={ordered_set.predecessor(random_elem_a)}")
    print(f"Testing Successor of [{random_elem_b}]={ordered_set.successor(random_elem_b)}")

    set_b = OrderedSet(str)
    for i in data_b:
        set_b.add(i)

    print(f"Testing Subset: Is Set A a subset of set B? (all elements of set a must be in set b)={ordered_set.subset(set_b)}")

    print(f"\nTesting Union: Combines two sets together. removing any duplicates: ")
    new_set = ordered_set.union(set_b)
    print(f"result: {new_set}")

    print(f"\nTesting Intersection:")
    inter_set = ordered_set.intersection(set_b)
    print(f"{inter_set}")
    print(f"\nTesting Difference:")
    diff_set = ordered_set.difference(set_b)
    print(f"{diff_set}")
    print(f"\nTesting Symmetric Difference")
    symm_diff_set = ordered_set.symmetric_difference(set_b)
    print(f"{symm_diff_set}")

    random_elem_c = random.choice(list(ordered_set))
    print(f"\nTesting Split: seperator element = {random_elem_c}")
    print(f"{ordered_set}")
    split_set_a, split_set_b = ordered_set.split(random_elem_c)
    print(f"set a = {split_set_a}")
    print(f"set b = {split_set_b}")

    print(f"\nTesting Join")

if __name__ == "__main__":
    main()
