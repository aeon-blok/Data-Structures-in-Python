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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
# endregion

# region custom imports
from types.custom_types import T

# endregion


"""
Collection ADT: 

All common ADTs are derived from the collection ADT - which defines a minimal, universal interface.
it is sometimes known as the Container ADT or Aggregate ADT
"""

class CollectionADT(Generic[T]):
    """ Minimal Universal Interface for Data Structures"""

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

    
    # ----- Optional ADT Operations -----

    # @abstractmethod
    # def __iter__(self) -> Generator[T, None, None]:
    #     """Iterates over all the elements in the sequence - used in loops and ranges etc"""
    #     pass
