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


LoadFactor = NewType("LoadFactor", float)   # New type from float for - domain logic
BitMask = NewType("BitMask", int)

class ValidateLoadFactor:
    """A float between 0 and 1. -- used to measure the Load Factor of Hash Tables"""
    def __new__(cls, value: float):
        if not isinstance(value, (float, int)):
            raise DsTypeError(f"Error: Invalid Type:  Expected: {float.__name__} Got: {str(type(value))}. Load Factor must be a Float type.")

        value = float(value)    # convert to float (if the input is an int like 0)

        if value is None:
            raise DsUnderflowError(f"Error: Load Factor cannot be a None value.")
        if not 0 <= value < 1:
            raise DsInputValueError(f"Error: Load Factor Value must be between 0.0 and 1.0")

        return LoadFactor(value)


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
