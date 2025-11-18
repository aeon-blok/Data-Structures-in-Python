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
    Iterable,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence
import math

# endregion


# region custom imports
from utils.custom_types import T, K, Key
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray

# endregion


class HashFunctions:
    def __init__(self, map_obj) -> None:
        self.obj = map_obj

    # -------------------------------- Utilities   --------------------------------

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

    # -------------------------------- Compression Functions --------------------------------

    def _k_mod_compression_function(self, hash_code, table_capacity):
        """Takes a hash code and conforms it to the hash table size, and returns the index number"""
        # the division method: aka k-mod
        k_mod = hash_code % table_capacity
        return k_mod

    def _mad_compression_function(self, hash_code):
        """The MAD Method - multiply - add - divide method: Takes a hashcode and conforms to table capacity - returns the index number"""
        # M-A-D Method core logic
        multiply = self.scale * hash_code
        add = multiply + self.shift
        divide = add % self.prime
        index = divide % self.capacity  # finally mod by table capacity
        return index

    def _second_hash_function(self, key):
        """creates a simple second hash function for step size for double hashing"""
        second_hash_code = self._cyclic_shift_hash_code(key)
        return 1 + (second_hash_code % (self.capacity - 1))
    

    # -------------------------------- Hash Codes  --------------------------------

    def _polynomial_hash_code(self, key):
        """polynomial hash code: uses Horners Method"""
        prime_weighting = 33    # small prime number: commonly 33, 37, 39, 41 - we will randomize and initialize on hashtable creation
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character)
        return hash_code

    def _cyclic_shift_hash_code(self, key, shift:int = 7):
        """Cyclic Shift Hash Code: uses bitwise shifting"""
        word_bit_size = 64
        bit_mask = 2**64-1  # This creates a 64-bit mask
        hash_code = 0
        for char in key:
            # word_bit_size & bit_mask masks the result to 256 bits, effectively discarding any higher bits.
            hash_code = ((hash_code << shift) | (hash_code >> (word_bit_size - shift))) & bit_mask
            hash_code = hash_code ^ ord(char)
        return hash_code

    def _cyclic_polynomial_combo_hash_code(self, key, shift: int = 7):
        prime_weighting = 33  # small prime number: commonly 33, 37, 39, 41 - we will randomize and initialize on hashtable creation
        bit_mask = 2**64 - 1  # This creates a 64-bit mask
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character) & bit_mask
        hash_code ^= (hash_code << shift) & bit_mask
        hash_code ^= hash_code >> shift
        hash_code ^= hash_code << (shift // 2) & bit_mask

        return hash_code & bit_mask

    def _hash_function(self, key: str):
        """Combines the hash code and compression function and returns an index value for a key."""
        poylnomial_hashcode = self._polynomial_hash_code(key)   # better for smaller tables like up to 1000 (0 collisions)
        cyclic_shift_hashcode = self._cyclic_shift_hash_code(key)   # better for huge tables like 10,000+ (1000 collisions)

        index = self._mad_compression_function(cyclic_shift_hashcode)
        return index
