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
from user_defined_types.generic_types import T


# endregion


"""
Deque ADT:

"""


class DequeADT(ABC, Generic[T]):
    """
    Deque ADT Interface: Canonical operations
    """
    # ----- Canonical ADT Operations -----

    # ----- Accessor ADT Operations -----
    @property
    @abstractmethod
    def front(self) -> Optional[T]:
        pass

    @property
    @abstractmethod
    def rear(self) -> Optional[T]:
        pass

    # ----- Mutator ADT Operations -----

    @abstractmethod
    def add_front(self, element: T) -> None:
        """Add an Element to the Front of the Deque"""
        pass

    @abstractmethod
    def add_rear(self, element: T) -> None:
        """Add an Element to the Back of the Deque"""
        pass

    @abstractmethod
    def remove_front(self) -> Optional[T]:
        """Remove an element from the front of the deque"""
        pass

    @abstractmethod
    def remove_rear(self) -> Optional[T]:
        """remove an element from the back of the deque"""
        pass
