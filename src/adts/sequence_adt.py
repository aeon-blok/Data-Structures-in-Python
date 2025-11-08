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
    Iterable
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
# endregion

# region custom imports
from utils.constants import T

# endregion


# array adt

"""
Sequence ADT: collection of elements of type E in linear order

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


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
        """Replace element at index i with value"""
        pass

    @abstractmethod
    def insert(self, index, value: T):
        """Insert value at index i, shift elements right"""
        pass

    @abstractmethod
    def delete(self, index: int) -> T:
        """Remove element at index i, shift elements left"""
        pass

    @abstractmethod
    def append(self, value: T):
        """Add value at end N-1"""
        pass

    @abstractmethod
    def prepend(self, value: T):
        """Insert value at index 0"""
        pass

    @abstractmethod
    def index_of(self, value: T) -> Optional[int]:
        """Return index number of first value (if exists)"""
        pass

    # ----- Optional ADT Operations -----
    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        pass
