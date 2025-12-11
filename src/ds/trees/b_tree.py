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
from user_defined_types.generic_types import (
    T,
    ValidDatatype,
    TypeSafeElement,
    Index,
    ValidIndex,
)
from utils.validation_utils import DsValidation
from utils.representations import BTreeNodeRepr, BTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.b_tree_adt import BTreeADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_nodes import BTreeNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal

# endregion

"""
B Tree:
"""


class BTree(BTreeADT[T], CollectionADT[T], Generic[T]):
    """
    B Tree Data Structure Implementation:
    """
    def __init__(self, datatype: type) -> None:
        self._datatype = datatype
        self._keytype: None | type = None
        self._root = None

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = BTreeRepr(self)

    # ----- Meta Collection ADT Operations -----

    def is_empty(self) -> bool:
        return super().is_empty()

    def clear(self) -> None:
        return super().clear()

    def __contains__(self, value: T) -> bool:
        return super().__contains__(value)

    def __len__(self) -> Index:
        return super().__len__()

    def __iter__(self):
        pass

    def __reversed__(self):
        pass

    # ----- Utilities -----

    def __str__(self) -> str:
        return super().__str__()
    
    def __repr__(self) -> str:
        return super().__repr__()
    
    def __bool__(self):
        pass

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def search(self, key) -> T | None:
        return super().search(key)

    def predecessor(self, key):
        return super().predecessor(key)

    def successor(self, key):
        return super().successor(key)

    def min(self):
        return super().min()

    def max(self):
        return super().max()

    # ----- Mutators -----

    def insert(self, key, value: T) -> None:
        return super().insert(key, value)

    def delete(self, key) -> None:
        return super().delete(key)

    # ----- Traversal -----

    def traverse(self) -> Iterable[Tuple]:
        return super().traverse()

    # ----- Utility -----

    def split_child(self, parent_node, index) -> None:
        return super().split_child(parent_node, index)

    def merge_children(self, parent_node, index) -> None:
        return super().merge_children(parent_node, index)

    def borrow_left(self, parent_node, index) -> None:
        return super().borrow_left(parent_node, index)

    def borrow_right(self, parent_node, index) -> None:
        return super().borrow_right(parent_node, index)
