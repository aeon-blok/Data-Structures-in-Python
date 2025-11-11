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

def index_boundary_check(index: int, capacity: int, is_insert: bool = False) -> None:
    """Checks that the index is a valid number for the array. -- index needs to be greater or equal to 0 and smaller than the number of elements (size)"""
    if is_insert:
        if index < 0 or index > capacity:
            raise IndexError("Error: Index is out of bounds.")
    else:
        if index < 0 or index >= capacity:
            raise IndexError("Error: Index is out of bounds.")
