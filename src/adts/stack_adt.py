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


"""
Dynamic array based Stack. Automatically resizes when it gets close to full.
Dynamic capacity (resize up/down, usually ×2 / ÷2).
Double capacity on full 
Half capacity when ≤25% full
user supplied initial capacity > 1
Static Type Validation
Overflow & Underflow Errors
"""


T = TypeVar("T")


# interface
class StackADT(ABC, Generic[T]):
    """Stack ADT - defines the necessary methods for a stack"""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def push(self, value: T) -> None:
        pass

    @abstractmethod
    def pop(self) -> T:
        pass

    @abstractmethod
    def peek(self) -> T:
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass
