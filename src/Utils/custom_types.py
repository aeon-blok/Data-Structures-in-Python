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


# These are reused everywhere. That’s what “types” is for.


# Custom Types

T = TypeVar("T")  # generic type
K = TypeVar("K")  # Keys usually must be hashable.
V = TypeVar("V")  # Values can be anything.


Predicate = Callable[[T], bool] # represents a function that returns a boolean 
HashFunction = Callable[[K], int]   # takes a key, and returns an int
