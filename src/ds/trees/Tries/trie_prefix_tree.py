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
from utils.representations import TrieRepr
from utils.helpers import RandomClass
from utils.exceptions import *
from utils.constants import PAGE_SIZE, ALPHABET

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
from user_defined_types.tree_types import NodeColor, Traversal, PageID

# endregion


class Trie(TrieADT[T], CollectionADT[T], Generic[T]):
    """Trie Data Structure Implementation using Hash Map for children node."""
    def __init__(self) -> None:
        self._root = TrieNode(None)
        self._alphabet = HashSet(str)
        for i in ALPHABET: self._alphabet.add(i)

        # composed objects
        self._utils = TreeUtils(self)
        self._validators = DsValidation()
        self._desc = TrieRepr(self)

    # ----- Meta Collection ADT Operations -----

    def __len__(self) -> Index:
        return super().__len__()

    def __contains__(self, value: T) -> bool:
        return super().__contains__(value)

    def is_empty(self) -> bool:
        return super().is_empty()

    def clear(self) -> None:
        return super().clear()

    def __iter__(self):
        pass

    # ----- Utilities -----
    def _prune(self, node) -> None:
        return super()._prune(node)

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----
    def search(self, key) -> bool:
        return super().search(key)

    def starts_with_prefix(self, prefix) -> bool:
        return super().starts_with_prefix(prefix)

    def get_node(self, prefix):
        return super().get_node(prefix)

    def enumerate(self, prefix):
        return super().enumerate(prefix)

    def height(self) -> Index:
        return super().height()

    def children(self, node):
        return super().children(node)

    def has_children(self, node) -> bool:
        return super().has_children(node)

    # ----- Mutators -----
    def insert(self, key: str) -> bool:
        """Inserts a New Key into the Trie (usually a string(word)) -- O(M) (M=length of the string)"""
        
        # type check
        if not isinstance(key, str):
            raise DsTypeError(f"Error: Key must be a string! key={type(key)}")
        
        # * validate every character in the key via the alphabet
        for char in key:
            if char not in self._alphabet:
                raise DsInputValueError(f"Error: Character={char} does not exist in trie validation set: (alphabet)")

        # * traverse trie
        current = self._root

        for char in key:
            # * create a new node if it doesnt exist in the trie.
            if char not in current.children:
                current.children[char] = TrieNode(char)
            current = current.children[char]

        # if after traversing through tree - all the characters exist, the key already exists in the trie, return false.
        if current.is_end: 
            return False
        
        # * signify end of key / word
        current.is_end = True
        
        return True


        

    def delete(self, key) -> bool:
        return super().delete(key)


# ------------------------------- Main: Client Facing Code: -------------------------------
def main():
    pass


if __name__ == "__main__":
    main()
