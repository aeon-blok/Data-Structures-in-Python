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

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.trees.tree_utils import TreeUtils

from user_defined_types.generic_types import (
    Index,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    PositiveNumber,
)

from user_defined_types.tree_types import NodeColor, Traversal, PageID, SegmentOperator

# endregion


class LazySumSegmentTree():
    """
    Segment Tree that uses lazy propagation to increment multiple elements in a range at the same time. (RANGE_ADD)
    """
    def __init__(self, input_array: Sequence[int]) -> None:
        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = SegmentTreeRepr(self)

        self.array: Sequence = input_array
        self.array_length = len(self.array)
        self.comparator = SegmentOperator.SUM.desc
        self.merge = SegmentOperator.SUM.func

        # dummy value for initializing tree array. we must overwrite these before returning results.
        self.dummy_value = self._utils.get_dummy_value(SegmentOperator.SUM)
        self.tree = [self.dummy_value] * (4*self.array_length)
        self.build_segment_tree()
        self.lazy = [0] * (4 * self.array_length)  # a cache that stores pending increments (for range_add)

    # ----- Utilities -----
    @property
    def operator_type(self):
        return self.comparator

    def __len__(self) -> int:
        """this provides the size of the original input array."""
        return self.array_length

    @property
    def tree_size(self) -> int:
        """this returns the total number of nodes or elements in the segment tree (array)"""
        return len(self.tree)

    def __str__(self) -> str:
        return self._desc.str_lazy_segment_tree()

    def __repr__(self) -> str:
        return self._desc.repr_segment_tree()

    # ----- Canonical ADT Operations -----

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
        self._recursive_build(2 * index + 1, left, mid)
        # recursively build right subtree
        self._recursive_build(2 * index + 2, mid + 1, right)

        # * create parent node: merge 2 child segment nodes together to create the parent node
        self.tree[index] = self.merge(
            self.tree[2 * index + 1], self.tree[2 * index + 2]
        )

    def build_segment_tree(self) -> None:
        """constructs a segment tree from an input array. the resulting tree is represented as an array also."""

        # existence check
        if self.array_length == 0:
            return

        # the segment is the entire input array. left = first element, right = last element.
        left = 0
        right = self.array_length - 1
        self._recursive_build(0, left, right)

    def _apply_add(self, index, value, segment_length):
        """
        Apply value (sum, min, max) to this entire segment without applying the value to children nodes.
        the change is added to the lazy cache so it can propagate when needed.
        """
        # Case 1: SUM: Incrementing each element increases sum by value * length
        self.tree[index] += value * segment_length
        # records result in lazy cache: All children of this node are conceptually +value behind.
        self.lazy[index] += value

    def _push(self, index, left, right):
        """
        Apply any pending lazy update at the specified index - this will be applied to the children of specified index
        You must push before: descending during updates or descending during queries
        """

        # exit condition: no pending update → nothing to do or leaf → nowhere to push
        if left == right or self.lazy[index] == 0:
            return

        # Pushes a node’s pending lazy update down one level to its children.
        mid = (left + right) // 2
        self._apply_add(2*index+1, self.lazy[index], mid-left+1)
        self._apply_add(2*index+2, self.lazy[index], right-mid)
        # lazy cache at index reset to default state.
        self.lazy[index] = 0

    def _recursive_range_add(self, index, left, right, query_left, query_right, value):
        """recursive method that updates the query range by the value. will increment the already existing values."""

        # query range does not overlap. no further action needed
        if query_right < left or right < query_left:
            return

        # Update the node if its within valid query range boundaries.
        if query_left <= left and right <= query_right:
            self._apply_add(index, value, right-left+1)
            return

        # pushes cached operation down to children
        self._push(index, left, right)

        # divide and conquer - recursively apply method and we can resolve all the children nodes.
        mid = (left + right) // 2
        self._recursive_range_add(2*index+1, left, mid, query_left, query_right, value)
        self._recursive_range_add(2*index+2, mid+1, right, query_left, query_right, value)

        # combine children to form parent node.
        self.tree[index] = self.merge(self.tree[2*index+1], self.tree[2*index+2])

    def range_increment(self, left, right, value):
        """public method - increments every element in the range by the value"""
        self._utils.validate_query_range(left, right)
        self._recursive_range_add(0, 0, self.array_length-1, left, right, value)

    def _recursive_range_query(self, index, left, right, query_left, query_right):
        """returns the query value - respects lazy propaggation."""

        # query not found
        if query_right < left or right < query_left:
            return self.dummy_value

        # - no computation needed - query value exists and is located
        if query_left <= left and right <= query_right:
            return self.tree[index]

        # lazy propagate - before we recurse into children nodes. (ensures they have accurate values.)
        self._push(index, left, right)

        # recursive divide and conquer
        mid = (left + right) // 2
        left_result = self._recursive_range_query(2*index+1, left, mid, query_left, query_right)
        right_result = self._recursive_range_query(2*index+2, mid+1, right, query_left, query_right)
        return self.merge(left_result, right_result)

    def range_query(self, left, right):
        """Public Method --  queries a specific range of values."""
        self._utils.validate_query_range(left, right)
        return self._recursive_range_query(0,0,self.array_length-1, left, right)

    def point_update(self, index, value):
        """Updates a single element."""
        current = self.range_query(index, index)
        self.range_increment(index, index, value-current)


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():
    test_data = [i for i in range(10)]

    seg_tree = LazySumSegmentTree(test_data)
    print(repr(seg_tree))
    print(seg_tree)
    print(f"Query Range Test: {seg_tree.range_query(5,8)}")
    print(f"Testing Range Add - incrementing a range of values.")
    seg_tree.range_increment(0,5,30)
    print(seg_tree)
    print(f"Testing Point Update of a single element in the array. ")
    print(test_data)
    seg_tree.point_update(0, 200)
    print(seg_tree)

if __name__ == "__main__":
    main()
