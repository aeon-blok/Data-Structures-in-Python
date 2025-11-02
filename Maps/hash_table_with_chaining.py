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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random

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
T = TypeVar('T')


# Interfaces
class MapADT(ABC, Generic[T]):
    """Contains the Canonical Operations defined by the Map ADT"""

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def put(self, key: str, value: T):
        """Insert a key value pair into the hash table: if the key already exists we return the existing value"""
        pass

    @abstractmethod
    def get(self, key: str, default: Optional[Any]) -> T:
        """retrieves a key value pair from the hash table, with an optional default if the key is not found."""
        pass

    @abstractmethod
    def remove(self, key: str):
        """removes a key value pair from the hash table."""
        pass

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Does the Hash table contain an item with the specified key?"""
        pass

    @abstractmethod
    def keys(self) -> str:
        """Return a set of all the keys in the hash table"""
        pass

    @abstractmethod
    def values(self):
        """Return a set of all the values in the hash table"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Returns the number of key-value pairs in the hash table"""
        pass

    # ----- Meta Collection ADT Operations -----
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


CTYPES_DATATYPES = {
    int: ctypes.c_int,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    str: ctypes.py_object,  # arbitrary Python object (strings)
    object: ctypes.py_object,  # any Python object
}

NUMPY_DATATYPES = {
    int: numpy.int32,
    float: numpy.float64,
    bool: numpy.bool_,
}

class BucketArray(Generic[T]):
    """dynamic array - used in hash table. uses ctype array and ctype objects (general accept any python object). grow the array dynamically when full"""
    def __init__(self) -> None:
        pass


class ChainHashTable(MapADT[T]):
    """Hash Table implementation with collision chaining via array bucket."""
    def __init__(self, table_capacity: int, bucket_capacity: int, max_load_factor: float, ) -> None:

        if table_capacity <= 0:
            table_capacity = 10

        if max_load_factor <= 0 or max_load_factor >= 1:
            max_load_factor = 0.65

        # core attributes
        self.max_load_factor = max_load_factor  # prevents the table from exceeding this capacity
        self.current_size = 0   # number of key-value pairs
        self.table_capacity = table_capacity    # number of slots in hash table
        self.bucket_capacity = bucket_capacity  # number of slots in each bucket (each slot in the hash table)
        self.buckets = None
        # MAD attributes
        self.prime = self._find_next_prime_number(self.table_capacity)
        self.scale = random.randint(2, self.prime-1)
        self.shift = random.randint(1, self.prime-1)


    # ----- Utility -----

    def _is_prime_number(self, number):
        """Boolean Check if number is a prime."""
        if number < 2:
            return False
        for i in range(2, int(math.isqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    def _find_next_prime_number(self, table_capacity):
        """Finds the next prime number larger than the current table capacity."""
        candidate = table_capacity + 1
        while True:
            if self._is_prime_number(candidate):
                return candidate
            candidate += 1

    def _mad_compression_function(self, hash_code):
        """The MAD Method - multiply - add - divide method: Takes a hashcode and conforms to table capacity - returns the index number"""
        # M-A-D
        multiply = self.scale * hash_code
        add = multiply + self.shift
        divide = add % self.prime
        index = divide % self.table_capacity  # finally mod by table capacity
        return index

    def _k_mod_compression_function(self, hash_code):
        """Takes a hash code and conforms it to the hash table size, and returns the index number"""
        # the division method: aka k-mod
        k_mod = hash_code % self.table_capacity
        return k_mod

    def _polynomial_hash_code(self, key):
        """polynomial hash code: uses Horners Method"""
        prime_weighting = 33    # small prime number: commonly 33, 37, 39, 41
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character)
        return hash_code

    def _cyclic_shift_hash_code(self, key, shift:int = 5):
        """Cyclic Shift Hash Code: uses bitwise shifting"""
        word_bit_size = 256
        bit_mask = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        hash_code = 0
        for char in key:
            # word_bit_size & bit_mask masks the result to 256 bits, effectively discarding any higher bits.
            hash_code = ((hash_code << shift) | (hash_code >> (word_bit_size - shift))) & bit_mask
            hash_code = hash_code ^ ord(char)
        return hash_code

    def _hash_function(self, key):
        """Combines the hash code and compression function and returns an index value for a key."""

        
    def _rehash_table(self):
        """dynamically resizes the hash table when it reaches a certain capacity defined by max load factor."""
        pass

    # ----- Canonical ADT Operations -----
    def put(self, key, value):
        """
        Insert a key value pair into the hash table:
        if the key already exists we return the existing value
        Compute the hash of the key
        Find the bucket for the key
        Add the key-value pair to the bucket array.
        """
        pass

    def get(self, key, default=None):
        """
        Compute the hash of the key
        Look for the key in the bucket.
        Return the value if found.
        """
        pass

    def remove(self, key):
        """
        Compute the hash of the key.
        Look in the bucket array for the key
        Remove the key-value pair if found.
        and return the value
        """
        pass

    def contains(self, key):
        """Does the Hash table contain an item with the specified key?"""
        pass

    def keys(self):
        """Return a set of all the keys in the hash table"""
        pass

    def values(self):
        """Return a set of all the values of the hash table"""
        pass

    def items(self):
        """Returns a set of tuples of all the key value pairs in the hash table"""

    def size(self):
        """Returns the total number of key value pairs currently stored in the table"""
        pass

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        pass

    def clear(self):
        """Remove all key value pairs from the hash table."""
        pass

    def __iter__(self):
        pass
