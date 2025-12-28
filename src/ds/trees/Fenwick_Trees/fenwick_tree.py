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
from utils.representations import FenwickTreeRepr
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


class SumFenwickTree:
    """
    Fenwick Tree Data Structure Implementation: Used for range sum queries over an array of data.
    This implementation calculates the sums of the specified ranges of the array.
    indexing is 1-Based: via 
    """
    def __init__(self, array_length: int) -> None:
        self.array_length = array_length
        # index 0 is unused. this is critical for the correct functioning of a Fenwick tree.
        self.tree = [0] * (array_length + 1)  # stores the sum of indices 

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = FenwickTreeRepr(self)

    def __str__(self) -> str:
        return self._desc.str_fenwick_tree()

    def __repr__(self) -> str:
        return self._desc.repr_fenwick_tree()

    # ----- Canonical ADT Operations -----

    # ----- Mutators -----
    def build_fenwick_tree(self, input_array: Sequence[int]):
        """
        precomputes all cached partial sums instead of discovering them via updates.
        Use this when:
        You already have the full array
        You don’t need per-element update calls
        You want fastest initialization
        """
        for i in range(1, self.array_length+1):
            self.tree[i] += input_array[i]

            parent = i + (i & -i)

            if parent <= self.array_length:
                self.tree[parent] += self.tree[i]

    def update(self, index, value):
        """
        increments the value at the specified index.
        updates all the connected implicit nodes that are related to this index.
        """
        self._utils.validate_fenwick_tree_index(index)
        # walks up the implicit tree.
        while index <= self.array_length:
            self.tree[index] += value
            # jumps to the next parent node
            index += index & -index  # Isolates the lowest set bit, this Gives the node size

    # ----- Accessors -----
    def calculate_prefix_sum(self, index):
        """returns the prefix sum from index 1 (not 0) to the specified index"""

        if index < 0 or index > self.array_length:
            raise DsInputValueError("Query index out of bounds")

        running_sum_total = 0

        # walk down the implicit tree and collect the sum.
        while index > 0:
            running_sum_total += self.tree[index]
            index -= index & -index
        return running_sum_total
   
    def range_query(self, left, right):
        """Public method -- returns the sum of the specified range."""
        self._utils.validate_fenwick_tree_query_range(left, right)
        return self.calculate_prefix_sum(right) - self.calculate_prefix_sum(left-1)


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():
    test_data = [0] + [random.randint(i, 100) for i in range(10)]
    print(test_data)
    fenwick = SumFenwickTree(len(test_data)-1)
    fenwick.build_fenwick_tree(test_data)
    print(fenwick)
    print(repr(fenwick))



if __name__ == "__main__":
    main()
