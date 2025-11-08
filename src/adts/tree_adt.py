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


"""
Tree ADT: A tree is a hierarchical data structure consisting of nodes with a parent - child relationship.

Tree Operations:

Accessor Operations:

Mutator Operations:

Traversal Operations

Properties:
There is only one root node.
No cycles are allowed. acyclic
Every node has 1 parent.
Root node is None
All nodes are reachable from the root

Nodes:
there are several types of Nodes: root, parent, child, sibling & leaf

Terminology:
Depth: - the number of children a node has.
Height: the length in terms of nodes of the longest path from root to leaf node.

"""

T = TypeVar("T")


class TreeADT(ABC, Generic[T]):
    """Tree"""

    # ----- Canonical ADT Operations -----

    # ----- Accessors -----
    @property
    @abstractmethod
    def root(self) -> T:
        """Returns the value of the Root Node"""
        pass

    @abstractmethod
    def parent(self, node) -> Optional["iNode[T]"]:
        """returns the parent NODE of a specified node"""
        pass

    @abstractmethod
    def child(self, node) -> Optional["iNode[T]"]:
        """returns the child NODE of a specified node"""
        pass

    @abstractmethod
    def num_children(self, node) -> int:
        """returns the total number of children of a specified node"""
        pass

    @abstractmethod
    def is_root(self, node) -> bool:
        """returns true if the node is the root of a tree"""
        pass

    @abstractmethod
    def is_leaf(self, node) -> bool:
        """returns True if the node is a leaf node (no children)"""
        pass

    @abstractmethod
    def is_internal(self, node) -> bool:
        """returns True if the node has children nodes."""
        pass

    @abstractmethod
    def size(self) -> int:
        """returns total number of nodes in the tree"""
        pass

    @abstractmethod
    def depth(self, node) -> int:
        """returns Number of edges from the ROOT down to the specified node"""
        pass

    @abstractmethod
    def height(self, node) -> int:
        """returns Max Number of edges from a specified node to a leaf node (no children)."""
        pass

    # ----- Mutators -----
    @abstractmethod
    def createTree(self, value: T) -> "iNode[T]":
        """creates a new tree with a root node"""
        pass

    @abstractmethod
    def addChild(self, parent_node, value) -> "iNode[T]":
        """adds a child node to the specified node."""
        pass

    @abstractmethod
    def remove(self, node) -> "iNode[T]":
        """removes a specified node and all its descendants"""
        pass

    @abstractmethod
    def replace(self, node, value) -> "iNode[T]":
        """replaces a value in a specified node"""
        pass

    # ----- Traversals -----
    @abstractmethod
    def preorder(self) -> Optional[list[T]]:
        """Depth First Search: (DFS) -- travels from root to last child - returns a list of values"""
        pass

    @abstractmethod
    def postorder(self) -> Optional[list[T]]:
        """Depth First Search: (DFS) travels from last child to root - returns a list of values"""
        pass

    @abstractmethod
    def level_order(self) -> Optional[list[T]]:
        """Breadth First Search: (BFS) --- visiting nodes level by level, - starts from left -> right, and traverses the entire tree top -> bottom"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        """returns total number of nodes in the tree"""
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass


# Node interface
class iNode(ABC, Generic[T]):
    """interface for Tree ADT node"""

    @property
    @abstractmethod
    def value(self) -> T:
        """return the value stored inside the node"""
        pass

    @property
    @abstractmethod
    def parent(self) -> Optional["iNode[T]"]:
        """return the parent node or None if this is the root"""
        pass

    @property
    @abstractmethod
    def children(self) -> Optional[list["iNode[T]"]]:
        """return a list of all the children nodes"""
        pass

    # ----- Mutators -----
    @abstractmethod
    def add_child(self, value: T) -> None:
        """insert a child under this node"""
        pass

    @abstractmethod
    def remove_child(self, node: "iNode[T]") -> T:
        """removes a specific child node"""

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

    # maybe include size and depth here also.
