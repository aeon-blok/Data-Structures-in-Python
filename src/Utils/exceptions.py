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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


class DsError(Exception):
    """Base class for all Data Structures exceptions."""
    pass

class DsOverflowError(DsError):
    """Error when a data structure exceeds its capacity. (its full)"""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: The Data Structure is full."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class DsUnderflowError(DsError):
    """Error when a data structure is empty."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: The Data Structure is empty."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class DsTypeError(DsError):
    """Error when datatype enforcement fails."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Invalid Type"
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeEmptyError(DsError):
    """Error when Node Object returns None"""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Node Object cannot be None."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeTypeError(DsError):
    """Error when Node reference is the wrong type (must be a Node type  - defined by iNode interface)"""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Node Reference is an Invalid Type"
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeDeletedError(DsError):
    """Error when a linked list node has already been deleted from a list, but user is attempting to reference it."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Reference Node was deleted and is no longer linked to the list."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeOwnershipError(DsError):
    """Error when Node reference doesnt belong to the instantiated list."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Reference Node does not belong to this list."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeExistenceError(DsError):
    """Error when Node does not exist."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Node doesnt exist..."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)

class NodeNotFoundError(DsError):
    """Error when Node not found in List."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Node was not found in the list..."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)


class KeyInvalidError(DsError):
    """Error when the key provided to the list or dictionary is of an invalid type."""
    def __init__(self, message: Optional[str] = None) -> None:
        self._default_message = f"Error: Invalid Key Provided. Please use a Node or an Index Number."
        self._message = self._default_message if message is None else message
        super().__init__(self._message)
