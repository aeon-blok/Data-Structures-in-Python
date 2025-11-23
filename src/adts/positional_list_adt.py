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
    TYPE_CHECKING
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T
from adts.collection_adt import CollectionADT


# endregion

"""
Positional List ADT:
A Positional List ADT is not the same as a linked list but very similar.
It Adds an abstraction & encapsulation layer ontop of Linked list logic in order to decouple the end use from the underlying logic.
It does this via a position object. An obect that utilizes the proxy pattern to represent nodes in the positional list (1 for each node). All end user interaction is done through position objects.
A Positional List MUST allow for bidirectional traversal.
This allows for efficient insertion and deletetion in constant time O(1)

Properties:
Position Based Abstraction:
Bidirectional Navigation:
Element Position Binding:
Stable Ordering:
Homogenous:
Finite:
Mutability:
No random index access

Constraints:
Position Validity:
O(1) Common Operations
Unique Identity For Position Objects
Boundary Safety
Encapsulation
Order Integrity
"""

class PositionalListADT(ABC, Generic[T]):
    """ADT for Positional List -- Contains all Canonical Operations - Guarantees O(1) insertion, replace, deletion..."""

    # ----- Accessor ADT Operations -----

    @abstractmethod
    def first(self) -> Optional["iPosition"]:
        """return the head position (position 0)"""
        pass

    @abstractmethod
    def last(self) -> Optional["iPosition"]:
        """return the tail position (position N-1)"""
        pass

    @abstractmethod
    def before(self, position: "iPosition[T]") -> Optional["iPosition"]:
        """return the position BEFORE specified position"""
        pass

    @abstractmethod
    def after(self, position: "iPosition[T]") -> Optional["iPosition"]:
        """return the position AFTER specified position"""
        pass

    @abstractmethod
    def get(self, position: "iPosition[T]") -> T:
        """Return element(data) from specified position"""
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """iterates through the elements"""
        pass

    # ----- Mutator ADT Operations -----
    @abstractmethod
    def add_first(self, element: T) -> Optional["iPosition[T]"]:
        """Insert an element value and a new position at the Head"""
        pass

    @abstractmethod
    def add_last(self, element: T) -> Optional["iPosition[T]"]:
        """Insert an element value & new position at the Tail"""
        pass

    @abstractmethod
    def add_before(self, position: "iPosition[T]", element: T) -> Optional["iPosition[T]"]:
        """Insert an element value & position before a specified position"""
        pass

    @abstractmethod
    def add_after(self, position: "iPosition[T]", element: T) -> Optional["iPosition[T]"]:
        """Insert an element value & position after a specified position"""
        pass

    @abstractmethod
    def replace(self, position: "iPosition[T]", element: T) -> T:
        """Replaces the element(value) at a specified Position"""
        pass

    @abstractmethod
    def delete(self, position: "iPosition[T]") -> T:
        """Removes an element(value) & position also. it reorganizes the list to not have these items."""
        pass


class iNode(ABC, Generic[T]):
    """Internal Class Interface for Positional List Nodes. """

    @property
    @abstractmethod
    def prev(self)-> Optional["iNode[T]"]:
        pass

    @property
    @abstractmethod
    def next(self) -> Optional["iNode[T]"]:
        pass

    @property
    @abstractmethod
    def element(self) -> T:
        pass


class iPosition(ABC, Generic[T]):
    """External Interface for Proxy Object for the Nodes."""

    @property
    @abstractmethod
    def element(self) -> T:
        pass
