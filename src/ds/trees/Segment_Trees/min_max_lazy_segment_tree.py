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
import sys

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


class LazyMinMaxSegmentTree:
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
        self.comparator = SegmentOperator.MAX.desc
        self.merge = SegmentOperator.MAX.func

        self.min_array = [sys.maxsize] * (4 * self.array_length)
        self.max_array = [-sys.maxsize] * (4 * self.array_length)
        self.lazy = [0] * (4 * self.array_length)
        self.build_segment_tree()

    # ----- Utilities -----
    @property
    def operator_type(self):
        return f"{self.comparator} + {SegmentOperator.MIN.desc}"

    def __len__(self) -> int:
        """this provides the size of the original input array."""
        return self.array_length

    @property
    def tree_size(self) -> int:
        """this returns the total number of nodes or elements in the segment tree (array)"""
        return len(self.min_array)

    def __str__(self) -> str:
        return self._desc.str_lazy_minmax_segment_tree()

    def __repr__(self) -> str:
        return self._desc.repr_segment_tree()

    # ----- Canonical ADT Operations -----

    # ----- Mutators -----

    def _pull(self, index):
        """"""
        self.min_array[index] = min(self.min_array[index*2+1], self.min_array[index*2+2])
        self.max_array[index] = max(self.max_array[index*2+1], self.max_array[index*2+2])

    def _apply(self, index, value):
        self.min_array[index] += value
        self.max_array[index] += value
        self.lazy[index] += value

    def _push(self, index):
        if self.lazy[index] != 0:
            self._apply(index*2+1, self.lazy[index])
            self._apply(index*2+2, self.lazy[index])
            self.lazy[index] = 0

    def _recursive_build(self, index, left, right):
        """recursive build helper method"""
        if left == right:
            self.min_array[index] = self.max_array[index] = self.array[left]
            return

        mid = (left + right) // 2
        self._recursive_build(index*2+1, left, mid)
        self._recursive_build(index*2+2, mid+1, right)
        self._pull(index)

    def build_segment_tree(self) -> None:
        """constructs a segment tree from an input array. the resulting tree is represented as an array also."""

        # existence check
        if self.array_length == 0:
            return

        # the segment is the entire input array. left = first element, right = last element.
        left = 0
        right = self.array_length - 1
        self._recursive_build(0, left, right)

    def _recursive_range_add(self, index, left, right, query_left, query_right, value):
        """recursively increments the elmements in the specified range, by the specified amount"""
        # query range does not overlap. no further action needed
        if query_right < left or right < query_left:
            return
        #
        if query_left <= left and right <= query_right:
            self._apply(index, value)
            return

        # * get accurate children before recursing
        self._push(index)

        # * divide & conquer
        mid = (left + right) // 2
        if query_left <= mid:
            self._recursive_range_add(index*2+1, left, mid, query_left, query_right, value)
        if query_right > mid:
            self._recursive_range_add(index*2+2, mid+1, right, query_left, query_right, value)
        #
        self._pull(index)

    def range_increment(self, left, right, value):
        """Public Update Range Method: allows to increase the nodes that fall within a specified range by a certain value"""
        if self.array_length == 0:
            return None

        self._recursive_range_add(0,0,self.array_length-1, left, right, value)

    def _recursive_range_min(self, index, left, right, query_left, query_right):
        """"""
        # query range does not overlap. no further action needed
        if query_right < left or right < query_left:
            return sys.maxsize
        # already stores the correct min for this segment
        if query_left <= left and right <= query_right:
            return self.min_array[index]
        # If this node has a pending range-add: propagate to children. ensures child values are up to date
        self._push(index)
        mid = (left + right) // 2
        dummy = sys.maxsize
        if query_left <= mid:
            dummy = min(dummy, self._recursive_range_min(index*2+1, left, mid, query_left, query_right) )
        if query_right > mid:
            dummy = min(dummy, self._recursive_range_min(index*2+2, mid+1, right, query_left, query_right))
        return dummy

    def _recursive_range_max(self, index, left, right, query_left, query_right):
        """recursively """
        # query range does not overlap. no further action needed
        if query_right < left or right < query_left:
            return -sys.maxsize

        # already stores the correct maximum for this segment
        if query_left <= left and right <= query_right:
            return self.max_array[index]

        self._push(index)
        mid = (left + right) // 2
        dummy = -sys.maxsize
        if query_left <= mid:
            dummy = max(dummy, self._recursive_range_max(index*2+1, left, mid, query_left, query_right))
        if query_right > mid:
            dummy = max(dummy, self._recursive_range_max(index*2+2, mid+1, right, query_left, query_right))
        return dummy

    def query_min_range(self, left, right):
        """Public Query Method: """
        return self._recursive_range_min(0,0,self.array_length-1, left, right)

    def query_max_range(self, left, right):
        """Public Query Method: """
        return self._recursive_range_max(0,0,self.array_length-1,left, right)

    def point_update(self, orig_array_index, value):
        """Updates a single implicit node / element -- from the original array"""
        self.range_increment(orig_array_index, orig_array_index, value)


# ------------------------------- Main: Client Facing Code: -------------------------------


def main():
    test_data = [i for i in range(10)]

    seg_tree = LazyMinMaxSegmentTree(test_data)
    print(repr(seg_tree))
    print(seg_tree)
    print(f"Query Range Test for Min Values: {seg_tree.query_min_range(5,8)}")
    print(f"Query Range Test for Max Values: {seg_tree.query_max_range(4,9)}")

    print(f"Testing Range Add - incrementing a range of values.")
    seg_tree.range_increment(0, 5, 30)
    print(seg_tree)
    print(f"Testing Point Update of a single element in the array. ")
    print(test_data)
    seg_tree.point_update(0, 200)
    print(seg_tree)


if __name__ == "__main__":
    main()
