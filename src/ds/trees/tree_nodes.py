# region standard lib
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
import uuid
from pprint import pprint
# endregion

# region custom imports
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import TreeNodeRepr, BinaryNodeRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.tree_adt import iTNode
from adts.binary_tree_adt import iBNode

if TYPE_CHECKING:
    from adts.tree_adt import TreeADT
    from adts.binary_tree_adt import BinaryTreeADT
    


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_utils import TreeNodeUtils
# endregion


class TNode(iTNode[T], Generic[T]):
    """Node for general tree implementaiton"""
    def __init__(self, datatype: type, element: T, tree_owner: "TreeADT[T]" | iTNode[T] | None = None) -> None:

        self._datatype = datatype
        self._element: T = element
        self._parent: Optional[iTNode[T]] = None
        self._children: List[iTNode[T]] = []
        self._tree_owner: "TreeADT[T]" | iTNode[T] | None = tree_owner
        self._deleted: bool = False

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc = TreeNodeRepr(self)

        self._validators.check_input_value_exists(self._element)
        self._validators.validate_datatype(self._datatype)

    @property
    def deleted(self) -> bool:
        return self._deleted

    @deleted.setter
    def deleted(self, value: bool):
        self._deleted = value

    @property
    def alive(self) -> bool:
        return not self._deleted

    @property
    def tree_owner(self):
        return self._tree_owner

    @tree_owner.setter
    def tree_owner(self, value):
        self._tree_owner = value

    @property
    def datatype(self):
        return self._datatype

    @property
    def element(self) -> T:
        return self._element

    @element.setter
    def element(self, value: T):
        self._element = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = value

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_tnode()

    def __repr__(self) -> str:
        return self._desc.repr_tnode()

    # ----- Mutators -----
    def add_child(self, element):
        """insert a child under this node"""
        self._validators.enforce_type(element, self._datatype)
        new_node = TNode(self._datatype, element, tree_owner=self._tree_owner)
        new_node.parent = self
        self._children.append(new_node)
        return new_node

    def remove_child(self, node):
        """
        removes a specific child node
        Step 1: Store node for return
        Step 2: unlink child - remove from children list
        Step 3: traverse child node subtree - and dereference all nodes
        Step 4: return node value
        """
        self._utils.validate_tnode(node)
        deleted_node = node
        deleted_value = node._element
        node._tree_owner = None
        node._deleted = True

        subtree = [node]  # reference subtree in a list(stack)

        # removes node from children list.
        self._children.remove(node)

        # dereference children.
        # By the end of this loop, the entire subtree is disconnected and ready for garbage collection.
        while subtree:
            current_node = subtree.pop()
            subtree.extend(current_node.children)
            # dereference nodes - both children and parent
            current_node.children = []
            current_node._tree_owner = None
            current_node.parent = None
            current_node._deleted = True

        return deleted_value

    # ----- Accessors -----
    def num_children(self):
        """returns the total number of children of a specified node -- ONLY counts direct children."""
        return len(self._children)

    def is_root(self):
        """returns true if the node is the root of a tree"""
        return self._parent is None

    def is_leaf(self):
        """returns True if the node is a leaf node (no children)"""
        return len(self._children) == 0

    def is_internal(self):
        """returns True if the node has children nodes."""
        return len(self._children) > 0

    # -------------- Testing Node Solo Functionality -----------------


class BinaryNode(iBNode[T], Generic[T]):
    """Node for a Basic Binary Tree"""
    def __init__(self, datatype: type, element: T, tree_owner=None) -> None:
        self._datatype = datatype
        self._element = element
        self._parent = None
        self._left = None
        self._right = None
        self._tree_owner = tree_owner
        self._deleted: bool = False

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc = BinaryNodeRepr(self)

        self._validators.check_input_value_exists(self._element)
        self._validators.validate_datatype(self._datatype)
        self._validators.enforce_type(self._element, self._datatype)

    @property
    def datatype(self):
        return self._datatype

    @property
    def element(self):
        return self._element
    @element.setter
    def element(self, value):
        self._element = value

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def left(self):
        return self._left
    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right
    @right.setter
    def right(self, value):
        self._right = value

    @property
    def sibling(self):
        """derived from the parent if it exists."""
        # no parent case:
        if self._parent is None:
            return None
        # check parent left child if its this node, its sibling must be right.
        if self.parent.left is self:
            return self.parent.right
        else:
            return self.parent.left

    @property
    def tree_owner(self):
        return self._tree_owner
    @tree_owner.setter
    def tree_owner(self, value):
        self._tree_owner = value

    @property
    def deleted(self):
        return self._deleted
    @deleted.setter
    def deleted(self, value):
        self._deleted = value

    @property
    def alive(self):
        return not self._deleted

    # ----- Utilities -----

    def __str__(self) -> str:
        return self._desc.str_binary_node()

    def __repr__(self) -> str:
        return self._desc.repr_binary_node()

    # ----- Accessors -----
    def num_children(self) -> int:
        counter = 0
        if self._left:
            counter += 1
        if self._right:
            counter += 1
        return counter    

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return self.num_children() == 0

    def is_internal(self) -> bool:
        return self.num_children() > 0


class BSTNode(iBNode[T], Generic[T]):
    """Node for a Basic Binary Tree"""

    def __init__(self, datatype: type, element: T, tree_owner=None) -> None:
        self._datatype = datatype
        self._element = element
        self._parent = None
        self._left = None
        self._right = None
        self._tree_owner = tree_owner
        self._deleted: bool = False

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc = BinaryNodeRepr(self)

        self._validators.check_input_value_exists(self._element)
        self._validators.validate_datatype(self._datatype)
        self._validators.enforce_type(self._element, self._datatype)

    @property
    def datatype(self):
        return self._datatype

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    @property
    def sibling(self):
        """derived from the parent if it exists."""
        # no parent case:
        if self._parent is None:
            return None
        # check parent left child if its this node, its sibling must be right.
        if self.parent.left is self:
            return self.parent.right
        else:
            return self.parent.left

    @property
    def tree_owner(self):
        return self._tree_owner

    @tree_owner.setter
    def tree_owner(self, value):
        self._tree_owner = value

    @property
    def deleted(self):
        return self._deleted

    @deleted.setter
    def deleted(self, value):
        self._deleted = value

    @property
    def alive(self):
        return not self._deleted

    # ----- Utilities -----

    def __str__(self) -> str:
        return self._desc.str_binary_node()

    def __repr__(self) -> str:
        return self._desc.repr_binary_node()

    # ----- Accessors -----
    def num_children(self) -> int:
        counter = 0
        if self._left:
            counter += 1
        if self._right:
            counter += 1
        return counter

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return self.num_children() == 0

    def is_internal(self) -> bool:
        return self.num_children() > 0


def main():
    node_a = TNode(str, "NODE ROOT")
    print(repr(node_a))
    print(node_a)

    child_a = node_a.add_child("new String to test")
    child_b = node_a.add_child("woatttt are you saying mate?")
    child_bb = child_b.add_child("ill fuck you up....")
    print(node_a.children)
    print(f"Number of direct children for node_a: {node_a.num_children()}")
    removed = node_a.remove_child(child_a)
    print(child_bb)
    print(f"Testing Parent property: {child_bb.parent}")
    print(f"Testing is_root: {child_b.is_root()}")
    print(f"Testing is_leaf: {child_b.is_leaf()}")
    print(f"Testing is_internal: {node_a.is_internal()}")
    print(child_b)
    print(repr(child_b))

if __name__ == "__main__":
    main()
