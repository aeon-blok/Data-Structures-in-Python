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
    TYPE_CHECKING
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
from utils.custom_types import T
# endregion


"""
Tree ADT: A tree is a hierarchical data structure consisting of nodes with a parent - child relationship.

Unique Path Invariant: for any node. there is only one path from root to the node.
Leaf Identity: Leaf nodes have no children
Parent Identity: Parent have 1 or more children.
Connectivity: Every node in the tree is reachable from the root.
Acyclicity: 

Properties:
There is only one root node.
No cycles are allowed. acyclic
Every node has 1 parent.
Root node is None
All nodes are reachable from the root

Terminology:
Depth: the number of children a node has.
Height: the length in terms of nodes of the longest path from root to leaf node.
Breadth: Number of Leaves(no children) attached to the tree.
Width: Number of nodes in a specific level.
level: the number of edges from root to a node. (all nodes are grouped by level in BFS)
Degree: The number of children it has (and number of subtrees attached to itself)
Nodes:there are several types of Nodes: root, parent, child, sibling & leaf
Edges: are the paths from one node to another. (connections between nodes)
Subtree: Is another tree that is connected to the main tree
"""


class TreeADT(ABC, Generic[T]):
    """Tree ADT:"""

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    @property
    @abstractmethod
    def root(self) -> T:
        """Returns the value of the Root Node"""
        pass

    @abstractmethod
    def parent(self, node: "iTNode[T]") -> Optional["iTNode[T]"]:
        """returns the parent NODE of a specified node"""
        pass

    @abstractmethod
    def children(self, node: "iTNode[T]") -> list["iTNode[T]"] | None:
        """returns the child NODE of a specified node"""
        pass

    @abstractmethod
    def num_children(self, node: "iTNode[T]") -> int:
        """returns the total number of children of a specified node"""
        pass

    @abstractmethod
    def is_root(self, node: "iTNode[T]") -> bool:
        """returns true if the node is the root of a tree"""
        pass

    @abstractmethod
    def is_leaf(self, node: "iTNode[T]") -> bool:
        """returns True if the node is a leaf node (no children)"""
        pass

    @abstractmethod
    def is_internal(self, node: "iTNode[T]") -> bool:
        """returns True if the node has children nodes."""
        pass

    @abstractmethod
    def depth(self, node: "iTNode[T]") -> int:
        """returns Number of edges from the ROOT down to the specified node"""
        pass

    @abstractmethod
    def height(self, node: "iTNode[T]") -> int:
        """returns Max Number of edges from a specified node to a leaf node (no children)."""
        pass

    # ----- Mutators -----
    @abstractmethod
    def createTree(self, value: T) -> "iTNode[T]":
        """creates a new tree with a root node"""
        pass

    @abstractmethod
    def addChild(self, parent_node: "iTNode[T]", value: T) -> "iTNode[T]":
        """adds a child node to the specified node."""
        pass

    @abstractmethod
    def remove(self, node: "iTNode[T]") -> "iTNode[T]":
        """removes a specified node and all its descendants"""
        pass

    @abstractmethod
    def replace(self, node: "iTNode[T]", value: T) -> "iTNode[T]":
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




# Node interface
class iTNode(ABC, Generic[T]):
    """interface for Tree ADT node"""

    @property
    @abstractmethod
    def value(self) -> T:
        """return the value stored inside the node"""
        pass

    @property
    @abstractmethod
    def parent(self) -> Optional["iTNode[T]"]:
        """return the parent node or None if this is the root"""
        pass

    @property
    @abstractmethod
    def children(self) -> Optional[list["iTNode[T]"]]:
        """return a list of all the children nodes"""
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

    # maybe include size and depth here also.
