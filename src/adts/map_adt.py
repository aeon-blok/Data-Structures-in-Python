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


# region custom imports
from user_defined_types.generic_types import T, K, iKey
if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from ds.primitives.arrays.dynamic_array import VectorArray


# endregion

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


# Interfaces
class MapADT(ABC, Generic[T, K]):
    """Contains the Canonical Operations defined by the Map ADT"""

    # ----- Canonical ADT Operations -----

    # ----- Mutators -----
    @abstractmethod
    def put(self, key: K, value: T) -> Optional[T]:
        """Insert a key value pair into the hash table: if the key already exists we return the existing value"""
        pass

    @abstractmethod
    def get(self, key: K, default: Optional[T]) -> Optional[T]:
        """retrieves a key value pair from the hash table, with an optional default if the key is not found."""
        pass

    @abstractmethod
    def remove(self, key: K) -> Optional[T]:
        """removes a key value pair from the hash table."""
        pass
    
    # ----- Accessors -----
    @abstractmethod
    def keys(self) -> Optional["VectorArray"]:
        """Return a set of all the keys in the hash table"""
        pass

    @abstractmethod
    def values(self) -> Optional["VectorArray"]:
        """Return a set of all the values in the hash table"""
        pass
