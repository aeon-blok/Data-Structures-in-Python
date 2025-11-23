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

from utils.exceptions import *
from types.custom_types import iKey


class DsValidation:
    """Global Validation Helper Methods for Data Structures"""
    def __init__(self) -> None:
        pass

    def validate_datatype(self, datatype):
        """ensures the datatype is a valid type"""
        if datatype is None:
            raise DsUnderflowError("Error: Datatype cannot be None Value.")
        if not isinstance(datatype, type):
            raise DsTypeError("Error: Datatype must be a valid Python Type object.")

    def enforce_type(self, value, expected_type):
        """type enforcement - checks that the value matches the prescribed datatype."""
        if not isinstance(value, expected_type):
            raise DsTypeError(f"Error: Invalid Type: Expected: [{expected_type.__name__}] Got: [{type(value)}]")

    def index_boundary_check(self, index: int, capacity: int, is_insert: bool = False) -> None:
        """Checks that the index is a valid number for the array. -- index needs to be greater or equal to 0 and smaller than the number of elements (size)"""
        if is_insert:
            if index < 0 or index > capacity:
                raise IndexError("Error: Index is out of bounds.")
        else:
            if index < 0 or index >= capacity:
                raise IndexError("Error: Index is out of bounds.")

    def check_input_value_exists(self, value):
        """Check input value exists...."""
        if value is None:
            raise DsInputValueError("Error: Must have an Input Value")

    def validate_key(self, key):
            """ensures the the input key, is a valid key."""
            if not isinstance(key, iKey):
                raise KeyInvalidError("Error: Input Key is not valid. All keys must be hashable, immutable & comparable (<, >, ==, !=)")
            elif key is None:
                raise KeyInvalidError("Error: Key cannot be None Value")
            return key