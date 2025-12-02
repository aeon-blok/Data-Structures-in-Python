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
    NewType,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
from enum import Enum, StrEnum, Flag, auto

# endregion

# region custom types

from utils.exceptions import *
from user_defined_types.key_types import iKey


# endregion


# region Atomic Types

T = TypeVar("T")  # generic type
K = TypeVar("K", bound='iKey')  # Keys usually must be hashable.
V = TypeVar("V")  # Values can be anything.


# New Types
Index = int   # alias for int

class ValidDatatype:
    """validates that a datatype is a valid type, and is not None."""
    def __new__(cls, value: type):
        if value is None:
            raise DsUnderflowError("Error: Datatype cannot be None Value.")
        if not isinstance(value, type):
            raise DsTypeError("Error: Datatype must be a valid Python Type object.")
        return value

class PositiveNumber(int):
    def __new__(cls, number: int):
        if number is None:
            raise DsInputValueError("Error: Index must not be None")
        if number < 0:
            raise DsInputValueError("Error: Number cannot be negative.")
        return number

class ValidIndex(int):
    """validate index number, ensure that it is positive number and in range of the capacity for the data structure."""
    def __new__(cls, index: int, capacity: int, array_insert: bool = False):
        if index is None:
            raise DsInputValueError("Error: Index must not be None")
        if array_insert:
            if index < 0 or index > capacity:
                raise IndexError("Error: Index is out of bounds.")
        else:
            if index < 0 or index >= capacity:
                raise IndexError("Error: Index is out of bounds.")
        return index

class TypeSafeElement:
    """ensures that the element matches the specified datatype."""
    def __new__(cls, value, datatype: type):
        if value is None:
            raise DsInputValueError("Error: Element must not be None at creation.")
        if not isinstance(value, datatype):
            raise DsTypeError(f"Error: Invalid Type: Expected: [{datatype.__name__}] Got: [{type(value).__name__}]")
        return value


# old protocol implementation
# class iKey(ABC):
#     """
#     Enforces that the type:
#     is comparable (<,>,==,!=
#     is hashable (compared for equality __eq__ -- can compare any object - required by base class)
#     """
#     @abstractmethod
#     def __lt__(self, other: "iKey") -> bool: ...
#     @abstractmethod
#     def __gt__(self, other: "iKey") -> bool: ...
#     @abstractmethod
#     def __eq__(self, other: object) -> bool:
#         """If two objects compare equal (__eq__), their hashes must also be equal."""
#         ...
#     @abstractmethod
#     def __hash__(self) -> int:
#         """
#         to be hashable, an object’s __hash__() method must return an integer.
#         Keys must be effectively immutable.
#         """
#         ...


# endregion


# region Combined Types


# endregion
