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
from user_defined_types.generic_types import T, K
from user_defined_types.key_types import Key, iKey
# endregion



weight =  object | None

class ValidVertex:
    """ensures that the element matches the specified datatype."""
    def __new__(cls, value, datatype: type):
        if value is None:
            raise NodeEmptyError("Error: Vertex Cannot be a None object. Please insert a valid Vertex Object.")
        if not isinstance(value, datatype):
            raise DsTypeError(f"Error: Invalid Type: Expected: [{datatype.__name__}] Got: [{type(value).__name__}]")
        return value
    

class VertexColor(StrEnum):
    """Colors for Vertex Graph Nodes"""
    WHITE = "white"
    GRAY = "gray"
    BLACK = "black"
