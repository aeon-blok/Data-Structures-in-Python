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
Disjoint Set ADT:
A Disjoint Set Data Structure Is a collection of non overlapping or unique collections of elements. These are usually represented by tree data structures and the implementation of a disjoint set is known as a disjoint forest
Properties:
    Partition Invariant: Every Element belongs to Only One Set.
    Every Set has exactly one representative.
    Find Operation is Idempotent.
    Disjoint Property: collections never overlap.
Constraints:
    Elements cannot be removed, once added. They stay together in the same set forever.
    Union Operation never Decreases the size of any existing set.
    Union Operation never splits a set.
    Find Operation always returns the same representative per set.
"""

class DisjointSetADT(ABC, Generic[T]):
    """
    ADT for Disjoint set
    typically disjoint sets utilize optimizations to vastly improve the Big O time complexity of standard operations, these are:
    Union by size / rank
    Path Compression
    """
    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @abstractmethod
    def find(self, element: T) -> Optional[T]:
        """Determines which set the input element belongs to, and returns the representative of that set (the root)"""
        pass

    # ----- Mutators -----
    @abstractmethod
    def make_set(self, element: T) -> None:
        """creates a new set with a single element. this will become the representative of the new set."""
        pass

    @abstractmethod
    def union(self, element_x: T, element_y: T) -> bool:
        """Merges the two sets containing x and y together to become one set. returns true if the union operations merged 2 sets together. returns false if the elements are already in the same set."""
        pass
    
    @abstractmethod
    def set_count(self) -> int:
        """returns the total number of disjoint sets in the data structure. (not the number of members for each set.)"""
        pass





