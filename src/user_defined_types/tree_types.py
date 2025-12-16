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

# type alias
PageID = int

class NodeColor(Enum):
    """Node Coloring Enum for Red Black Trees"""
    RED = "red"
    BLACK = "black"


class Traversal(StrEnum):
    """Tree Traversal Methods as Enum Types for Type Safety."""
    PREORDER = "preorder"
    POSTORDER = "postorder"
    LEVELORDER = "levelorder"
    INORDER = "inorder"


class ValidNode:
    """validates nodes and returns the original input."""
    def __new__(cls, node, node_datatype:type):
        if node is None:
            raise NodeEmptyError("Error: Node is None.")
        if not isinstance(node, node_datatype):
            raise DsTypeError("Error: Node is not a valid Node Type.")
        if not node.alive:
            raise NodeDeletedError(f"Error: Node has already been deleted.")
        return node

