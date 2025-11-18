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
    Protocol,
    runtime_checkable,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
from enum import Enum, StrEnum

# endregion


# Custom Types

T = TypeVar("T")  # generic type
K = TypeVar("K", bound='Key')  # Keys usually must be hashable.


V = TypeVar("V")  # Values can be anything.


Predicate = Callable[[T], bool] # represents a function that returns a boolean 

class HashCode(StrEnum):
    """Types for Hash Codes in one centralized place"""
    POLYNOMIAL = "polynomial"
    CYCLIC_SHIFT = "cyclic"
    POLYCYCLIC = "polycyclic"

class CompressFunc(StrEnum):
    """Compression Function Types"""
    MAD = "mad"
    KMOD = "kmod"
    DOUBLE_HASH = "doublehash"


@runtime_checkable  # ? allows the protocol to be used with isinstance() at runtime. (not automatic)
class Key(Protocol):
    """
    Enforces that the type:
    is comparable (<,>,==,!=
    is hashable (compared for equality __eq__ -- can compare any object - required by base class)
    """

    def __lt__(self, other: "Key") -> bool: ...
    def __gt__(self, other: "Key") -> bool: ...
    def __eq__(self, other: object) -> bool:
        """If two objects compare equal (__eq__), their hashes must also be equal."""
        ...

    def __hash__(self) -> int:
        """
        to be hashable, an object’s __hash__() method must return an integer.
        Keys must be effectively immutable.
        """
        ...
