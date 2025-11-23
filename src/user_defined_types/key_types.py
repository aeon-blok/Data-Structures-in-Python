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

# region custom imports
from utils.exceptions import *


# endregion


class iKey(ABC):
    """
    Enforces that the type:
    is comparable (<,>,==,!=
    is hashable (compared for equality __eq__ -- can compare any object - required by base class)
    """
    @abstractmethod
    def __lt__(self, other: "iKey") -> bool: ...
    @abstractmethod
    def __gt__(self, other: "iKey") -> bool: ...
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """If two objects compare equal (__eq__), their hashes must also be equal."""
        ...
    @abstractmethod
    def __hash__(self) -> int:
        """
        to be hashable, an object’s __hash__() method must return an integer.
        Keys must be effectively immutable.
        """
        ...
    @property
    @abstractmethod
    def value(self):
        pass

    @property
    @abstractmethod
    def datatype(self) -> type:
        pass


class Key(iKey):
    """Base Key class - validates that a key is hashable and the same type for comparisons. can be subclassed for further logic"""
    def __init__(self, value) -> None:
        if value is None: raise DsInputValueError(f"Error: Key cannot be a None value")
        try: hash(value)
        except TypeError as e: raise DsTypeError("Error: Key Type Must be Hashable.")
        self._value = value
        self._datatype = type(value)    # gets the datatype of the value for validation checks.

    @property
    def value(self):
        return self._value
    
    @property
    def datatype(self):
        return self._datatype

    def _assert_key_type(self, other):
        if not isinstance(other, iKey):
            raise KeyInvalidError(f"Error: Can only compare key with other key types.")
        return other
    
    def _assert_same_key_type(self, other):
        if self._datatype != other.datatype:
            raise DsTypeError(f"Error: Cannot compare keys of different datatypes. Ensure that you are using the same type.")
        return other

    def __repr__(self) -> str:
        return f"Key: {self._value!r}"
    
    def __lt__(self, other) -> bool:
        other = self._assert_key_type(other)
        other = self._assert_same_key_type(other)
        return self._value < other.value
    
    def __gt__(self, other) -> bool:
        other = self._assert_key_type(other)
        other = self._assert_same_key_type(other)
        return self._value > other.value
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, iKey):
            return False
        if self._datatype != other.datatype:
            return False
        return self._value == other.value
    
    def __hash__(self) -> int:
        return hash(self._value)
    

