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

"""
A Hash Table is a data structure that stores key-value pairs via a hash function which computes an index into an underlying bucket array.
This allows for fast insertion, lookup & deletion (O(1) average, O(N) worst)
Hash Collisions are handled via approaches like chaining or Open addressing

Axioms:
- Unique Keys: Each key in the map can store at most one value.
- Put & Get Consistency: If you `put(k, v)` into the map, a subsequent `get(k)` must return the same value `v`.
- Remove & Get Consistency: If you `remove(k)` from the map, a subsequent `get(k)` must return "undefined" (⊥)
- Contains & Get Equivalence: `contains(k)` returns true if and only if `get(k)` returns a value, not undefined.
- Invariant Keys: The set returned by 'keys()' always equals the entire domain of the map. That is - All keys that have values.
- Invariant Size: 'size()' always returns the total number of stored key-value pairs.
"""


# Custom Types
T = TypeVar("T")


# Interfaces
class MapADT(ABC, Generic[T]):
    """Contains the Canonical Operations defined by the Map ADT"""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def put(self, key: str, value: T):
        """Insert a key value pair into the hash table: if the key already exists we return the existing value"""
        pass

    @abstractmethod
    def get(self, key: str, default: Optional[T]) -> Optional[T]:
        """retrieves a key value pair from the hash table, with an optional default if the key is not found."""
        pass

    @abstractmethod
    def remove(self, key: str) -> Optional[T]:
        """removes a key value pair from the hash table."""
        pass

    @abstractmethod
    def keys(self) -> Optional["VectorArray"]:
        """Return a set of all the keys in the hash table"""
        pass

    @abstractmethod
    def values(self) -> Optional["VectorArray"]:
        """Return a set of all the values in the hash table"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of key-value pairs in the hash table"""
        pass

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Does the Hash table contain an item with the specified key?"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """The default iteration for a Map, is to generate a sequence (list) of all the keys in the map."""
        pass
