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
    Tuple,
    Literal,
    Iterable,
    TYPE_CHECKING,
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
from pprint import pprint

# endregion

# region custom imports
from types.custom_types import T

# endregion


"""
Binary Tree ADT:
"""


class BinaryTreeADT(ABC, Generic[T]):
    """Binary Tree ADT: specifies necessary canonical operations"""
    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @property
    @abstractmethod
    def root(self) -> Optional['iBNode[T]']:
        """returns the root NODE."""
        pass

    @abstractmethod
    def parent(self, node: "iBNode[T]") -> Optional['iBNode[T]']:
        """returns the parent node of the specified node."""
        pass

    @abstractmethod
    def left(self, node: "iBNode[T]") -> Optional['iBNode[T]']:
        """returns the left child node of the specified node"""
        pass

    @abstractmethod
    def right(self, node: "iBNode[T]") -> Optional['iBNode[T]']:
        """returns the right child node of the specified node"""
        pass

    @abstractmethod
    def num_children(self, node: "iBNode[T]") -> int:
        """returns the number of children of the specified node"""
        pass

    @abstractmethod
    def height(self) -> int:
        """returns the total height (max number of edges from root to furthest leaf) of the tree"""
        pass

    @abstractmethod
    def depth(self, node: "iBNode[T]") -> int:
        """returns the depth (number of edges from the root to this node.)"""
        pass

    # ----- Mutators -----
    @abstractmethod
    def add_root(self, element: T) -> Optional['iBNode[T]']:
        """adds the root node to the tree."""
        pass

    @abstractmethod
    def add_left(self, element: T, node: 'iBNode[T]') -> Optional['iBNode[T]']:
        """adds a left child node to the specified node."""
        pass

    @abstractmethod
    def add_right(self, element: T, node: 'iBNode[T]') -> Optional['iBNode[T]']:
        """adds a right child node to the specified node"""
        pass

    @abstractmethod
    def replace(self, element: T, node: "iBNode[T]") -> Optional[T]:
        """replaces the element value of the specified node."""
        pass

    @abstractmethod
    def delete(self, node: 'iBNode[T]') -> T:
        """deletes the specified node and reorganizes the tree."""
        pass


class iBNode(ABC, Generic[T]):
    # ----- Canonical ADT Operations -----

    @property
    @abstractmethod
    def element(self) -> T:
        """returns the value from this node"""
        pass

    @property
    @abstractmethod
    def parent(self) -> Optional["iBNode[T]"]:
        """returns the parent node of this node."""
        pass

    @property
    @abstractmethod
    def left(self) -> Optional['iBNode[T]']:
        """returns the left child of this node"""
        pass

    @property
    @abstractmethod
    def right(self) -> Optional['iBNode[T]']:
        """returns the right child of this node"""
        pass

    @property
    @abstractmethod
    def sibling(self) -> Optional["iBNode[T]"]:
        """returns the sibling of this node"""
        pass

    # ----- Accessors -----

    @abstractmethod
    def num_children(self) -> int:
        """returns the total number of children of a specified node"""
        pass

    @abstractmethod
    def is_root(self) -> bool:
        """returns true if the node is the root of a tree"""
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        """returns True if the node is a leaf node (no children)"""
        pass

    @abstractmethod
    def is_internal(self) -> bool:
        """returns True if the node has children nodes."""
        pass

    # ----- Mutators -----
