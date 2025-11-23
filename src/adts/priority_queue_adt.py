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
from user_defined_types.custom_types import T


# endregion

"""
Priority Queue ADT:
A Priority Queue allows you to access and only access the highest priority element in the data structure.


Invariants:
Ordering Invariant: extreme element is always globally correct
Strict Weak Ordering: Priorities must be:
    irreflexive: a priority cannot be less than itself (the identity (value) of a priority must remain consistent.)
    asymmetric: if a < b then never b < a (this is what allows comparison)
    Transitive: a < b and b < c implies a < c (transmutative)
    Incompatibility Transivity: if a is incomparable with b,c, then b is incomparable with c
Presence Invariant: element must exist for change-priority operations
Non Empty Invariant: Priority queue cannot be empty.
Consistency Invariant: Priority Queue must after every operation satisfy:
    All stored elements have a priority
    all priorities respect strict weak ordering
    the extreme element is correctly identifiable

Extreme Element Unique Identity Invariant: There can only be 1 element that qualifies as the extreme element.

Constraints:
Locality Irrelevance: Insertion order cannot influence which element is considered extreme. (only priority)
Stability Not Guaranteed: In case of a tie between priorities, the priority queue does NOT enforce FIFO behaviour (like a queue)
Element Identity requirements:
    Unique identity required
    unique reference for each element


how CLRS and Kleinberg/Tardos define it: one orientation per PQ instance.
"""


class MinPriorityQueueADT(ABC, Generic[T]):
    """
    Min Priority Queue ADT Interface: Canonical operations
    """

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----
    @abstractmethod
    def find_min(self) -> T:
        """equivalent to peek - returns but doesnt remove the top priority kv pair in the priority queue"""
        pass

    # ----- Mutator ADT Operations -----
    @abstractmethod
    def insert(self, element: T, priority: int) -> None:
        """add a key value pair to the Priority queue, while maintaining the order."""
        pass

    @abstractmethod
    def extract_min(self) -> Optional[T]:
        """remove and return the element with the top priority value"""
        pass

    @abstractmethod
    def decrease_key(self, element: T, priority: int) -> None:
        """Updates a specific elements - priority value."""
        pass


class MaxPriorityQueueADT(ABC, Generic[T]):
    """
    Max Priority Queue ADT Interface: Canonical operations
    """

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----
    @abstractmethod
    def find_max(self) -> T:
        """equivalent to peek - returns but doesnt remove the top priority kv pair in the priority queue"""
        pass

    # ----- Mutator ADT Operations -----
    @abstractmethod
    def insert(self, element: T, priority: int) -> None:
        """add a key value pair to the Priority queue, while maintaining the order."""
        pass

    @abstractmethod
    def extract_max(self) -> T:
        """remove and return the element with the top priority value"""
        pass

    @abstractmethod
    def increase_key(self, element: T, priority: int) -> None:
        """Updates a specific elements - priority value."""
        pass


class PriorityQueueADT(ABC, Generic[T]):
    """
    Generic Priority Queue ADT Interface:
    Can utilize a custom key to determine priority sorting behaviour
    """

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----


    @abstractmethod
    def find_extreme(self) -> T:
        """equivalent to peek - returns but doesnt remove the top priority kv pair in the priority queue"""
        pass

    # ----- Mutator ADT Operations -----
    @abstractmethod
    def insert(self, element: T, priority: int) -> None:
        """add a key value pair to the Priority queue, while maintaining the order."""
        pass

    @abstractmethod
    def extract_extreme(self) -> T:
        """remove and return the element with the top priority value (depends on the custom key behaviour -- can be min or max)"""
        pass

    @abstractmethod
    def change_priority(self, element: T, priority: int) -> None:
        """Updates a specific elements - priority value."""
        pass
