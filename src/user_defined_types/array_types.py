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
    Union,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import numpy.typing
import ctypes
from enum import Enum, StrEnum, Flag, auto

# endregion


# region custom imports
from utils.exceptions import *
from user_defined_types.generic_types import T


# endregion




class BSearch(StrEnum):
    """different search modes for binary search"""
    CLASSIC = "classic"
    RECURSIVE = "recursive"
    EXPONENTIAL = "exponential"
    INTERPOLATION = "interpolation"
    NOISY = "noisy"






