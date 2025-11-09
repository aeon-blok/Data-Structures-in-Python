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


def enforce_type(value, expected_type):
    """type enforcement - checks that the value matches the prescribed datatype."""
    if not isinstance(value, expected_type):
        raise TypeError(f"Error: Invalid Type: Expected: {expected_type.__name__} Got: {type(value)}")
