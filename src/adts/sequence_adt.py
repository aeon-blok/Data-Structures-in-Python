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


# array adt

"""
**Dynamic Array**: collection of elements of type E in linear order
A contiguous block of memory that resizes automatically when it runs out of space.

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


# Generic Type
T = TypeVar("T")


# interface
class SequenceADT(ABC, Generic[T]):
    """Sequence ADT: models an ordered, finite collection of elements, each accessible by an integer position (index)."""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def get(self, index) -> T:
        """Return element at index i"""
        pass

    @abstractmethod
    def set(self, index, value: T):
        """Replace element at index i with x"""
        pass

    @abstractmethod
    def insert(self, index, value: T):
        """Insert x at index i, shift elements right"""
        pass

    @abstractmethod
    def delete(self, index: int) -> T:
        """Remove element at index i, shift elements left"""
        pass

    @abstractmethod
    def append(self, value: T):
        """Add x at end N-1"""
        pass

    @abstractmethod
    def prepend(self, value: T):
        """Insert x at index 0"""
        pass

    @abstractmethod
    def index_of(self, value: T) -> Optional[int]:
        """Return index of first x (if exists)"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def __len__(self) -> int:
        """Return number of elements - formally defined as size()"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """returns true if sequence is empty"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """removes all items from the sequence"""
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        """True if x exists in sequence"""
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        pass
