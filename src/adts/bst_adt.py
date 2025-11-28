# region standard imports
from __future__ import annotations
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
from user_defined_types.generic_types import T, K

# endregion


"""
Binary Search Tree ADT: (BST)
all nodes are sorted via a key (which must be comparable (>,<,==, != etc))
BST ADT = Binary Tree ADT + ordering constraint:
You reuse your Binary Tree ADT exactly as-is (nodes, positions, parent, children).

Binary Search Property: (fundamental invariant)
This turns the tree into a decision structure, not a search space that must be fully scanned.

"""


class BinarySearchTreeADT(ABC, Generic[T, K]):
    """Binary Search Tree ADT: specifies the necessary operations"""
    # ----- Canonical ADT Operations -----
    # ----- Accessors -----

    @property
    @abstractmethod
    def root(self) -> Optional["iBSTNode[T, K]"]:
        """returns none or the root node"""
        pass

    @abstractmethod
    def parent(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the parent of specified node"""
        pass

    @abstractmethod
    def left(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the left child of the specified node"""
        pass

    @abstractmethod
    def right(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the right child of the specified node"""
        pass

    @abstractmethod
    def height(self) -> int:
        """returns the max edges from root to furthest leaf"""
        pass

    @abstractmethod
    def search(self, key: K) -> Optional["iBSTNode[T, K]"]:
        """returns the node that matches the specified key."""
        pass

    @abstractmethod
    def minimum(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the node with the lowest key in the subtree of the specified node."""
        pass

    @abstractmethod
    def maximum(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the node with the highest key in the subtree of the specified node."""
        pass

    @abstractmethod
    def successor(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the node with the Next higher key in the subtree of the specified node."""
        pass

    @abstractmethod
    def predecessor(self, node: "iBSTNode[T, K]") -> Optional["iBSTNode[T, K]"]:
        """returns the node with the Next Lower key in the subtree of the specified node."""
        pass

    # ----- Mutators -----
    @abstractmethod
    def insert(self, key: K, value: T) -> "iBSTNode[T, K]":
        """
        Inserts a new node into the the Binary Search Tree. 
        The tree automatically determines the correct spot based on key comparisons.
        """
        pass

    @abstractmethod
    def replace(self, node: 'iBSTNode[T, K]', value: T) -> T:
        """replaces the element value of the specified node. (no reorder necessary, the KEY didnt change)"""
        pass

    @abstractmethod
    def delete(self, node: iBSTNode[T, K]) -> T:
        """deletes a node from the BST tree. and returns the old element value"""
        pass


class iBSTNode(ABC, Generic[T, K]):
    """
    BST Node - has key property for comparisons and lookups. 
    (key property is hashable and comparable)
    """

    # ----- Canonical ADT Operations -----

    @property
    @abstractmethod
    def key(self) -> K:
        """returns the nodes key -- this is used to order the bst."""
        pass

    @property
    @abstractmethod
    def element(self) -> T:
        """returns the value from this node"""
        pass

    @property
    @abstractmethod
    def parent(self) -> Optional["iBSTNode[T, K]"]:
        """returns the parent node of this node."""
        pass

    @property
    @abstractmethod
    def left(self) -> Optional["iBSTNode[T, K]"]:
        """returns the left child of this node"""
        pass

    @property
    @abstractmethod
    def right(self) -> Optional["iBSTNode[T, K]"]:
        """returns the right child of this node"""
        pass

    @property
    @abstractmethod
    def sibling(self) -> Optional["iBSTNode[T, K]"]:
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
