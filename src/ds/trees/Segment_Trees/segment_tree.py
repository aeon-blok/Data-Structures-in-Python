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
import pickle
import os
import struct
from pathlib import Path
from faker import Faker
import logging
import logging.handlers
import traceback
import json

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
from utils.representations import SegmentTreeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.trie_adt import TrieADT

from ds.sequences.Stacks.array_stack import ArrayStack
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.hash_table_with_chaining import ChainHashTable
from ds.maps.Sets.hash_set import HashSet
from ds.trees.tree_nodes import TrieNode
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import (
    Index,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    PositiveNumber,
)

from user_defined_types.key_types import iKey, Key
from user_defined_types.tree_types import NodeColor, Traversal, PageID, SegmentOperator

# endregion


class SegmentTree():
    """
    Segment Tree Data Structure: Recursive Array Based Implementation
    """
    def __init__(self, input_array: Sequence[int], operator = SegmentOperator.SUM) -> None:
        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = SegmentTreeRepr(self)

        # dummy value for initializing tree array. we must overwrite these before returning results.
        self.dummy_value = self._utils.get_dummy_value(operator)

        self.array: Sequence = input_array
        self.array_length = len(self.array)
        self.merge = operator.value.__func__
        self.tree = VectorArray(4*self.array_length, int)
        for _ in range(4*self.array_length): self.tree.append(self.dummy_value)
        self.build_segment_tree()

    # ----- Utilities -----

    def __str__(self) -> str:
        return self._desc.str_segment_tree()

    def __repr__(self) -> str:
        return self._desc.repr_segment_tree()

    # ----- Canonical ADT Operations -----

    # ----- Accessors -----

    def _recursive_query(self, index, seg_left, seg_right, query_left, query_right):
        """
        Recursively computes the aggregate value over the intersection of:
        the current segment [seg_left, seg_right]
        the query range [query_left, query_right]
        """

        # * segment exists outside the query range. Return dummy value so it doesn’t affect merges
        if seg_right < query_left or seg_left > query_right:
            return self.dummy_value

        # * This segment is fully inside the query. return the value as is. recursion not necessary
        if query_left <= seg_left and seg_right <= query_right:
            return self.tree[index]

        # * divided and conquer - recursively check the children in the tree(array)
        mid = (seg_left + seg_right) // 2
        left = self._recursive_query(2*index+1, seg_left, mid, query_left, query_right)
        right = self._recursive_query(2*index+2, mid+1, seg_right, query_left, query_right)

        # * Merge child results to form the answer for this segment
        return self.merge(left, right)

    def query_range(self, left, right):
        """
        Returns the aggregate (sum / min / max / etc.) over array[l..r].
        """
        # * boundary check
        if left < 0 or right >= self.array_length or left > right:
            raise DsInputValueError("Error: Query range out of bounds of Segment Tree.")

        return self._recursive_query(0,0, self.array_length-1, left, right)

    # ----- Mutators -----
    def _recursive_build(self, index, left, right):
        """Compute the value for a segment [l, r] and store it in tree at index[i]."""

        # * Recursion Base Case: segment has only 1 element.
        if left == right:
            # The correct aggregate value of that segment is the element itself, So we store it directly
            self.tree[index] = self.array[left]
            return

        # * segment has more than 1 element. (divide & conquer)
        mid = (left + right) // 2
        # recursively build left subtree
        self._recursive_build(2*index+1, left, mid)
        # recursively build right subtree
        self._recursive_build(2*index+2, mid+1, right)

        # * create parent node: merge 2 child segment nodes together to create the parent node
        self.tree[index] = self.merge(self.tree[2*index+1], self.tree[2*index+2])

    def build_segment_tree(self) -> None:
        """constructs a segment tree from an input array. the resulting tree is represented as an array also."""

        # existence check
        if self.array_length == 0:
            return

        # the segment is the entire input array. left = first element, right = last element.
        left = 0
        right = self.array_length-1
        self._recursive_build(0, left, right)

    def _recursive_point_update(self, index, seg_left, seg_right, orig_array_index, element):
        """
        Recursively updates a single element in the original array and
        fixes all segment tree nodes whose segments include that index.
        """

        # * recursion base case
        if seg_left == seg_right:
            self.tree[index] = element
            return

        # * divide & conquer
        mid = (seg_left + seg_right) // 2
        if orig_array_index <= mid:
            self._recursive_point_update(2*index+1, seg_left, mid, orig_array_index, element)
        else:
            self._recursive_point_update(2*index+2, mid+1, seg_right, orig_array_index, element)

        # * after updating child nodes - rebuild parent via merge operation.
        self.tree[index] = self.merge(self.tree[2*index+1], self.tree[2*index+2])

    def point_update(self, orig_array_index, element):
        """Public Point Update Method: Updates the index and all connected nodes -- O(log n)"""

        # * boundary check
        if orig_array_index < 0 or orig_array_index >= self.array_length:
            raise DsInputValueError(f"Error: Index value is out of bounds.")
        
        # * update source array
        self.array[orig_array_index] = element

        # * recursively resolve all nodes that the original array node is connected to.
        self._recursive_point_update(0, 0, self.array_length-1, orig_array_index, element)
