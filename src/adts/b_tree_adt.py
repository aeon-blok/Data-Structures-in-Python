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
    Tuple,
    Literal,
    Iterable,
    TYPE_CHECKING,
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import T
from user_defined_types.key_types import iKey

# endregion


"""
B Tree ADT:
B-Tree is a specialized subclass of a multiway search tree
Its a Balanced Search Tree - but NOT a binary search tree.


Properties:
Automatic split on overflow and merge/borrow on underflow.

Invariants:
Root Invariant: Root contains 1 to (2t−1) keys (or 0 if empty tree).
Node Invariant: Every non-root node contains (t−1) to (2t−1) keys. (t is the minimum degree of the B-Tree. it must be >=2)
Children Invariant: A node with k keys has exactly k+1 children (if internal).
Subtree Order Invariant: all keys in the children must lie between the keys adjacent to that child.
Leaf Invariant: Leaves have no children
Sorted Order Invariant:  Keys inside each node appear in strictly increasing order.
Tree Depth Invariant:  Every leaf appears at the same depth.

Constraints: 

Splitting (Overflow rule):  
If insertion attempts to place a key into a node that already has 2t−1 keys, the node must split:
    Left child: t−1 keys
	Middle key: moves up to parent
	Right child: t−1 keys
	This happens recursively upward.

Merge or Borrow (Underflow rule): 
If deletion causes a node to fall below t−1 keys, it must be fixed by:
    Borrowing: a key from a sibling with ≥ t keys, via the parent.
    Merging: with a sibling (combining two children and one key from the parent).
"""


class BTreeADT(ABC, Generic[T]):
    """B Tree Adt - Supports all canonical operations"""

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @abstractmethod
    def search(self, key) -> Optional[T]:
        """
        Searches the B Tree for the specified key, returns element value
        """
        ...

    @abstractmethod
    def _predecessor(self, node) -> tuple:
        """returns the node and index of the largest key smaller than the specified key."""
        ...

    @abstractmethod
    def _successor(self, node) -> tuple:
        """returns the node and index of the smallest key larger than the specified key """
        ...

    @abstractmethod
    def min(self) -> Optional[T]:
        """returns the smallest key value pair in the B Tree"""
        ...

    @abstractmethod
    def max(self) -> Optional[T]:
        """returns the largest key value pair in the B Tree"""
        ...

    # ----- Mutators -----
    @abstractmethod
    def insert(self, key, value: T) -> None:
        """Inserts a key value pair into the B Tree - and maintains the B Tree invariants."""
        ...

    @abstractmethod
    def delete(self, key) -> None:
        """Deletes a key (and its value) from the B Tree while preserving B Tree invariants."""
        ...

        ...
    # ----- Utility -----
    @abstractmethod
    def split_child(self, parent_node, index) -> None:
        """Splits a Full Child node. Promotes the median key to the parent."""
        ...

    @abstractmethod
    def borrow_left(self, parent_node, idx) -> None:
        """Borrows a key value pair from a sibiling, via the parent"""
        ...

    @abstractmethod
    def borrow_right(self, parent_node, idx) -> None:
        """Borrows a key value pair from a sibiling, via the parent"""
        ...
