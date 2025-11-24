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
    Literal,
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


LoadFactor = float   # New type from float for - domain logic
BitMask = NewType("BitMask", int)
PercentageFloat = float
HashCode = int

class NormalizedFloat(float):
    """A float between 0 and 1. Represents a percentage."""
    def __new__(cls, value: float):
        if not isinstance(value, (float, int)):
            raise DsTypeError(f"Error: Invalid Type:  Expected: {float.__name__} Got: {str(type(value))}. Normalized Float must be a Float type.")
        value = float(value)    # convert to float (if the input is an int like 0)
        if value is None:
            raise DsUnderflowError(f"Error: Normalized Float cannot be a None value.")
        if not 0 <= value < 1:
            raise DsInputValueError(f"Error: Normalized Float Value must be between 0.0 and 1.0")
        return value

class ProbeType(StrEnum):
    """Types for Probe functions"""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    DOUBLE_HASH = "doublehash"
    PERTURBATION = "perturbation"
    RANDOM = "random"

class HashCodeType(StrEnum):
    """Types for Hash Codes in one centralized place"""
    POLYNOMIAL = "polynomial"
    CYCLIC_SHIFT = "cyclic"
    POLYCYCLIC = "polycyclic"

class CompressFuncType(StrEnum):
    """Compression Function Types"""
    MAD = "mad"
    KMOD = "kmod"

class Tombstone:
    """Tombstone Marker Class"""
    def __init__(self) -> None:
        pass
        
    def __str__(self) -> str:
        return f"🪦"
    
    def __repr__(self) -> str:
        return f"🪦"
