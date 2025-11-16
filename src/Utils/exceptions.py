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

from utils.helpers import Ansi

class DsError(Exception):
    """Base class for all Data Structures exceptions."""
    DEFAULT_ERROR_MESSAGE = "Error: An unexpected error occured..."
    COLOR = Ansi.RED

    def __init__(self, message: Optional[str]=None) -> None:
        message = message or self.DEFAULT_ERROR_MESSAGE
        message = Ansi.color(message, self.COLOR)
        super().__init__(message)


class DsOverflowError(DsError):
    """Error when a data structure exceeds its capacity. (its full)"""
    DEFAULT_ERROR_MESSAGE = f"Error: The Data Structure is full."


class DsUnderflowError(DsError):
    """Error when a data structure is empty."""
    DEFAULT_ERROR_MESSAGE = f"Error: The Data Structure is empty."


class DsDuplicationError(DsError):
    """Error when an element in a data structure already exists."""
    DEFAULT_ERROR_MESSAGE = f"Error: Element already exists."


class DsTypeError(DsError):
    """Error when datatype enforcement fails."""
    DEFAULT_ERROR_MESSAGE = f"Error: Invalid Type"


class DsInputValueError(DsError):
    """Error when Input Value is None"""
    DEFAULT_ERROR_MESSAGE = f"Error: Input Value cannot be None."


class NodeEmptyError(DsError):
    """Error when Node Object returns None"""
    DEFAULT_ERROR_MESSAGE = f"Error: Node Object cannot be None."

class NodeTypeError(DsError):
    """Error when Node reference is the wrong type (must be a Node type  - defined by iNode interface)"""
    DEFAULT_ERROR_MESSAGE = f"Error: Node Reference is an Invalid Type"


class NodeDeletedError(DsError):
    """Error when a linked list node has already been deleted from a list, but user is attempting to reference it."""
    DEFAULT_ERROR_MESSAGE = f"Error: Reference Node was deleted and is no longer linked to the list."

class NodeOwnershipError(DsError):
    """Error when Node reference doesnt belong to the instantiated list."""
    DEFAULT_ERROR_MESSAGE = f"Error: Reference Node does not belong to this list."

class NodeExistenceError(DsError):
    """Error when Node does not exist."""
    DEFAULT_ERROR_MESSAGE = f"Error: Node doesnt exist..."

class NodeNotFoundError(DsError):
    """Error when Node not found in List."""
    DEFAULT_ERROR_MESSAGE = f"Error: Node was not found in the list..."


class KeyInvalidError(DsError):
    """Error when the key provided to the list or dictionary is of an invalid type."""
    DEFAULT_ERROR_MESSAGE = f"Error: Invalid Key Provided. Please use a Node or an Index Number."

class PriorityInvalidError(DsError):
    """Error when the key provided to the list or dictionary is of an invalid type."""
    DEFAULT_ERROR_MESSAGE = f"Error: Priority input is invalid."
