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
from user_defined_types.generic_types import T, K
from utils.validation_utils import DsValidation
from utils.representations import TreeNodeRepr, BinaryNodeRepr, BSTNodeRepr, AVLNodeRepr
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.tree_adt import iTNode
from adts.binary_tree_adt import iBNode
from adts.bst_adt import iBSTNode

if TYPE_CHECKING:
    from adts.tree_adt import TreeADT
    from adts.binary_tree_adt import BinaryTreeADT
    from adts.bst_adt import BinarySearchTreeADT


from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.sequences.Stacks.array_stack import ArrayStack
from ds.trees.tree_utils import TreeNodeUtils

from user_defined_types.generic_types import ValidDatatype, TypeSafeElement
from user_defined_types.key_types import iKey, Key
# endregion

class BaseTreeNode(Generic[T]):
    """Base Tree Node Class to be inherited by other classes."""
    def __init__(self, datatype: type, element: T, tree_owner) -> None:
        self._datatype = ValidDatatype(datatype)
        self._element = TypeSafeElement(element, self.datatype)
        self._parent = None
        self._tree_owner = tree_owner
        self._alive: bool = True

    @property
    def alive(self) -> bool:
        return self._alive
    
    @alive.setter
    def alive(self, value):
        self._alive = value
        
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


class TNode(BaseTreeNode[T], iTNode[T], Generic[T]):
    """Node for general tree implementaiton"""
    def __init__(self, datatype, element, tree_owner = None) -> None:
        super().__init__(datatype, element, tree_owner) # base class inheritance

        self._children: List[iTNode[T]] = []

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc = TreeNodeRepr(self)

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
        self._utils.validate_tnode(node, iTNode)

        # store node for return
        deleted_node = node
        deleted_value = node._element

        # dereference node trackers
        node._tree_owner = None
        node._alive = False

        subtree = ArrayStack(iTNode)
        subtree.push(node)

        # removes node from children list.
        self._children.remove(node)

        # dereference children.
        # By the end of this loop, the entire subtree is disconnected and ready for garbage collection.
        while subtree:
            current_node = subtree.pop()
            for i in current_node.children:
                subtree.push(i)
            # dereference nodes - both children and parent
            current_node.children = []
            current_node._tree_owner = None
            current_node._parent = None
            current_node._alive = False
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


class BinaryNode(BaseTreeNode[T], iBNode[T], Generic[T]):
    """Node for a Basic Binary Tree"""
    def __init__(self, datatype, element, tree_owner=None) -> None:
        super().__init__(datatype, element, tree_owner)

        # Binary Node Unique Attributes 
        self._left = None
        self._right = None

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc = BinaryNodeRepr(self)

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
        if self.parent is None:
            return None
        # check parent left child if its this node, its sibling must be right.
        if self.parent.left is self:
            return self.parent.right
        else:
            return self.parent.left

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


class BSTNode(BaseTreeNode[T], iBSTNode[T, K], Generic[T, K]):
    """Node for a Basic Binary Tree"""
    def __init__(self, datatype: type, key: K, element: T, tree_owner=None) -> None:
        super().__init__(datatype, element, tree_owner)

        # BST Unique Attributes
        self._key = Key(key)
        self._left = None
        self._right = None

        # composed objects
        self._utils = TreeNodeUtils(self)
        self._validators = DsValidation()
        self._desc: BSTNodeRepr = BSTNodeRepr(self)

    @property
    def key(self):
        return self._key
    @key.setter
    def key(self, value):
        self._key = value

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
        if self.parent is None:
            return None
        # check parent left child if its this node, its sibling must be right.
        if self.parent.left is self:
            return self.parent.right
        else:
            return self.parent.left

    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_bst_node()

    def __repr__(self) -> str:
        return self._desc.repr_bst_node()

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


class AvlNode(BSTNode[T, K], Generic[T, K]):
    """Node for AVL trees - inherits from BST Node."""
    def __init__(self, datatype: type, key: K, element: T, tree_owner=None) -> None:
        super().__init__(datatype, key, element, tree_owner)
        # drives the rebalancing avl property. (modified after insertion / Deletion)
        self._height = 1
        self._avldesc: AVLNodeRepr = AVLNodeRepr(self)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def balance_factor(self) -> int:
        """the balance factor property - must be -1, 0 or 1 -- key feature of AVL trees"""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        return left_height - right_height

    def update_height(self):
        """Node has a self updating method. for the height property."""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = 1 + max(left_height, right_height)

    def __str__(self) -> str:
        return self._avldesc.str_avl_node()

    def __repr__(self) -> str:
        return self._avldesc.repr_avl_node()


# -------------- Testing Node Solo Functionality -----------------


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
