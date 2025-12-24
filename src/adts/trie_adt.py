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
    TYPE_CHECKING,
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
from user_defined_types.generic_types import T
from user_defined_types.key_types import iKey
from ds.trees.tree_nodes import TrieNode


"""
Trie ADT:
A prefix trie is an ordered tree data structure used in the representation of a set of strings over a finite alphabet set,
which allows efficient storage and retrieval of words with common prefixes
"""

class TrieADT(ABC):
    """contains all necessary operations for trie."""

    # ----- Canonical ADT Operations -----
    @property
    @abstractmethod
    def alphabet(self) -> Iterable:
        """a set of characters or digits used to validate keys and entries into the trie data structure."""
        ...

    # ----- Accessors -----
    @abstractmethod
    def search(self, word: str) -> bool:
        """Returns true if key exists and is the end of a word (is_end flag is set.)"""
        ...

    @abstractmethod
    def starts_with_prefix(self, prefix: str) -> bool:
        """Returns `true` if prefix path exists"""
        ...

    @abstractmethod
    def _get_node(self, prefix: str) -> Optional[TrieNode]:
        """Returns node representing the end of a prefix (or `null`)"""
        ...

    @abstractmethod
    def enumerate(self, prefix: str) -> Optional[Iterable]:
        """returns a list of all keys found for a specific prefix."""
        ...

    @abstractmethod
    def height(self) -> int:
        """returns the length of the longest key = the total tree height"""
        ...

    @abstractmethod
    def get_children(self, node) -> Iterable:
        """returns a list of all the children nodes for a specified node"""
        ...

    # ----- Mutators -----
    @abstractmethod
    def insert(self, word: str) -> bool:
        """Inserts a key and returns false if already present."""
        ...

    @abstractmethod
    def delete(self, word: str) -> bool:
        """removes a key and prunes the tree."""
        ...
