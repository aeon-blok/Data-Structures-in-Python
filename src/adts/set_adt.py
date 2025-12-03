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
# endregion


"""
Set ADT:
Properties:
    No Duplicates Allowed
    Unordered: elements can appear in any order
    Membership: We can check any element for membership in the set.
Two sets are equal if they contain the same elements.
Sets can be implemented via hash tables or binary search trees (red black / avl)
"""


class SetADT(ABC, Generic[T]):
    """The Canonical representation of a Set Data structure as an interface."""

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----

    @abstractmethod
    def subset(self, other: SetADT[T]) -> bool:
        """Checks if all the elements in this set are in set B"""
        ...

    # ----- Mutators -----
    @abstractmethod
    def add(self, element: T) -> None:
        """Inserts an element into the set, if not already present."""
        ...

    @abstractmethod
    def remove(self, element: T) -> None:
        """Removes an element from the set."""
        ...

    @abstractmethod
    def union(self, other: SetADT[T]) -> SetADT[T]:
        """Combines both sets together into a new set."""
        ...

    @abstractmethod
    def intersection(self, other: SetADT[T]) -> SetADT[T]:
        """Creates a new set, with the elements that are contained exclusively in both sets."""
        ...

    @abstractmethod
    def difference(self, other: SetADT[T]) -> SetADT[T]:
        """Creates a new set, with the elements that exist in set A but not Set B"""
        ...

    @abstractmethod
    def symmetric_difference(self, other: SetADT[T]) -> SetADT[T]:
        """returns a new set with elements in either set but not both"""
        ...
