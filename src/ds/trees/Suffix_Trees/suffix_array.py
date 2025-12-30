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
from utils.representations import SuffixArrayRepr
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


class SuffixArray:
    """
    Suffix Array Data structure: Utilizes Doubling Algorithm & Kaisai algorithm for building the suffix array.
    has LCP, LCS, LRS queries
    """
    def __init__(self, string_input: str) -> None:

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = SuffixArrayRepr(self)
        self.str = string_input
        self.str_length = len(self.str)
        self.suffix_array: list[int] = []
        self.rank: list[int] = []
        self.lcp: list[int] = []

        if string_input is not None:
            self.build_suffix_array()
            self.build_kasai_lcp_array()

    # ---------- utility ----------
    def __str__(self) -> str:
        return self._desc.str_suffix_array()
    
    def __repr__(self) -> str:
        return self._desc.repr_suffix_array()

    # ---------- build ----------
    def _manber_myers_doubling_algorithm(self) -> None:
        """Used for constructing the suffix array -- O(n log n)"""

        string_length = self.str_length  # Length of the string. Bounds suffix indices.
        # starts with 1 character and doubles every iteration -- If suffixes are correctly sorted by first k chars, we can sort by first 2k using two ranks.
        cur_prfx_len: int = 1
        # creates a List of suffix starting indices for each character in the string.
        suffix_array = list(range(self.str_length))
        # rank = ASCII value of character, Equivalent to sorting suffixes by first character.
        # We need a numeric rank to compare substrings efficiently.
        rank = [ord(char) for char in self.str]
        # Temporary array for next iteration’s ranks, Needed because rank updates must be based on previous ranks.
        next_rank = [0] * string_length 

        # * sorts the suffix indices by lexographic comparison, string comparison becomes int tuple comparison.
        # the key is a tuple (rank i, rank i+current_prefix_len)
        while True:
            # rank[i] is the lexicographic rank of substring
            suffix_array.sort(key=lambda i: (rank[i], rank[i + cur_prfx_len] if i + cur_prfx_len < string_length else -1))
            next_rank[suffix_array[0]] = 0  # the smallest suffix gets rank 0

            # We compare adjacent suffixes after sorting.
            for i in range(1, string_length):
                prev, current = suffix_array[i-1], suffix_array[i]
                # if the current suffix is larger = increment rank index.
                next_rank[current] = next_rank[prev] + ((rank[prev], rank[prev + cur_prfx_len] if prev + cur_prfx_len < string_length else -1) < (rank[current], rank[current+cur_prfx_len] if current + cur_prfx_len < string_length else -1))

            # replace old ranks with a copy of the newly computed ranks.
            rank = next_rank[:]

            # exit condition = If max rank is n-1, then: All ranks are unique & All suffixes fully ordered, no further action needed
            if rank[suffix_array[-1]] == string_length - 1:
                break

            # double the current prefix length
            cur_prfx_len <<= 1

        # store as instance attrs
        self.suffix_array = suffix_array
        self.rank = rank

    def build_suffix_array(self) -> None:
        """Public Method -- Builds the suffix array through a choice of algorithms."""
        self._manber_myers_doubling_algorithm()

    def build_kasai_lcp_array(self) -> None:
        """
        Utilizes the Kasai algorithm (O(n)) -- Kasai’s algorithm computes the LCP array in linear time, unlike naïve O(n²).
        Computes the length of longest common prefix between adjacent suffixes in lexicographic order.
        """
        str_length = self.str_length

        # string length too small
        if str_length < 2:
            self.lcp = []
            return

        lcp = [0] * (str_length - 1)
        cur_prfx_len = 0

        # Iterate over suffixes in text order, not suffix array order.
        for i in range(str_length):
            rank = self.rank[i]
            # exit condition: last suffix reached- reset cur_prfx
            if rank == str_length - 1:
                cur_prfx_len = 0
                continue
            # index of the suffix lexicographically after i
            next_idx = self.suffix_array[rank+1]
            # Compare characters starting from offset (current prefix length)
            while (
                i + cur_prfx_len < str_length and 
                next_idx + cur_prfx_len < str_length and 
                self.str[i+cur_prfx_len] == self.str[next_idx+cur_prfx_len]
                ):

                cur_prfx_len += 1   # traverse through
            # store the result
            lcp[rank] = cur_prfx_len
            # if cur prfx is not 0 decrement by 1
            if cur_prfx_len: cur_prfx_len -= 1

        # update instance attr
        self.lcp = lcp

    # ----- Accessors -----
    def search(self, str_suffix: str) -> bool:
        """binary search on suffix array"""
        left, right = 0, self.str_length-1

        while left <= right:
            mid = (left + right) // 2
            suffix = self.str[self.suffix_array[mid]:]
            if suffix.startswith(str_suffix):
                return True
            if suffix < str_suffix:
                left = mid + 1
            else:
                right = mid - 1
        return False

    def find_longest_common_prefix(self) -> int:
        """returns the length of the largest LCP"""
        return max(self.lcp) if self.lcp else 0

    def find_longest_repeated_substring(self):
        """
        A substring is repeated if it appears in at least two suffixes - this method finds the largest substring that appears in at least 2 suffixes.
        """

        # existence check
        if not self.lcp: return ""
        # length of the largest common prefix between any two adjacent suffixes
        i = self.lcp.index(max(self.lcp))
        start = self.suffix_array[i]
        return self.str[start: start + self.lcp[i]]

    def find_longest_common_substring(self, comparison_string: str):
        """
        finds the the longest contiguous block of characters that appears in both strings.
        """
        sep = chr(0)
        combined_string = self.str + sep + comparison_string
        split_idx = len(self.str)
        # create a new suffix array specifically with the combined string to solve this problem.
        new_suffix_array = SuffixArray(combined_string)
        new_suffix_array.build_suffix_array()
        new_suffix_array.build_kasai_lcp_array()

        # length of the best LCS found so far
        max_len: int = 0
        # starting index (in self.str) of that substring
        idx: int = 0

        # loops through suffixes (in lexographic order)
        for i in range(len(self.lcp)):
            a: int = new_suffix_array.suffix_array[i]
            b: int = new_suffix_array.suffix_array[i + 1]

            # Only consider suffix pairs from different strings
            # XOR comparison between two boolean conditions.
            if (a < split_idx) != (b < split_idx):
                if new_suffix_array.lcp[i] > max_len:
                    max_len = new_suffix_array.lcp[i]
                    idx = a

        # returns the combined string substring
        return combined_string[idx : idx + max_len]



# ------------------------------- Main: Client Facing Code: -------------------------------

def main():
    suffix = SuffixArray("undercomputative")
    print(f"{suffix}")
    print(f"Testing LCP: {suffix.find_longest_common_prefix()}")
    print(f"Testing LRS: {suffix.find_longest_repeated_substring()}")
    print(f"Testing LCS: {suffix.find_longest_common_substring('understand')}")


if __name__ == "__main__":
    main()
