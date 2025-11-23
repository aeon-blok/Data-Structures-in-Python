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
    Type,
    Iterable,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# region custom imports
from user_defined_types.generic_types import T


# endregion


"""
Stack ADT:
A stack is a sequential collection of elements that follow the LIFO principle (the last element inserted is also the first element to be removed)
Insertions and Deletions only occur at one position - the Top
"""


# interface
class StackADT(ABC, Generic[T]):
    """Stack ADT - defines the necessary methods for a stack"""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def push(self, element: T) -> None:
        """ Insert Element to the Top of the stack"""
        pass

    @abstractmethod
    def pop(self) -> T:
        """ remove and return the Top element of the stack"""
        pass

    @abstractmethod
    def peek(self) -> T:
        """ return (but do NOT remove) the Top element of the stack"""
        pass
