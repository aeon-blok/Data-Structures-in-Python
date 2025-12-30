# region standard imports
from __future__ import annotations
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
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
import random
import os, hashlib, math, itertools

# endregion

# region custom imports
from user_defined_types.generic_types import T, K
from user_defined_types.key_types import iKey
from utils.exceptions import *

from adts.set_adt import SetADT

# endregion


"""
Ordered Set ADT:
An Ordered Set Abstract Data Type represents a subset of a totally ordered universe of elements
It extends the Set ADT by preserving a global order on elements and supporting order-dependent operations.
In addition to standard set operations (add, remove, union, intersection, etc.),
it supports queries based on relative position and rank
    such as minimum/maximum, 
    predecessor/successor, 
    rank, 
    select, 

and structural operations like 
    split 
    join

The ADT specifies what these operations mean mathematically, not how they are implemented, and can be realized by data structures such as balanced search trees, treaps, skip lists, or B-trees.
"""


class OrderedSetADT(SetADT[T]):
    """Ordered Set ADT"""

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @abstractmethod
    def min(self) -> T:
        """
        Returns the element value from the smallest key in the Ordered Set.
        Sometimes this is called first()
        """
        ...

    @abstractmethod
    def max(self) -> T:
        """
        returns the element value from the largest key in the ordered set
        Sometimes this is called last()
        """
        ...

    @abstractmethod
    def predecessor(self, element: T) -> Optional[T]:
        """
        returns the Largest element that is smaller than the specified element.
        sometimes this is called previous()
        """
        ...

    @abstractmethod
    def successor(self, element: T) -> Optional[T]:
        """
        returns the smallest element that is larger than the specified element
        sometimes this is called next()
        """
        ...

    @abstractmethod
    def select_range(self, element_a: T, element_b: T) -> Optional[Iterable[T]]:
        """Returns a subset of all elements between two specified values"""
        ...

    # ----- Mutators -----
    @abstractmethod
    def split(self, seperator_element: T) -> tuple:
        """
        splits the ordered set into 2 seperate sets - all elements less than the specified element & all elements greater than the specified element
        also returns a boolean result if the specified element was actually in the ordered set or not.
        returns a tuple (set_a, boolean, set_b)
        """
        ...

    @abstractmethod
    def join(self, other: "OrderedSetADT[T]") -> OrderedSetADT[T]:
        """Merges 2 ordered sets into 1 - only if the largest element in set_a is smaller than the smallest element in set_b"""
        ...
