# region standard imports
from __future__ import annotations
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
from dataclasses import dataclass, field
# endregion


# region custom imports
from user_defined_types.generic_types import T, K
from utils.validation_utils import DsValidation
from utils.exceptions import *

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT
    from user_defined_types.key_types import iKey
    from user_defined_types.generic_types import Index

from user_defined_types.hashtable_types import (
    HashCode,
    HashCodeType,
    CompressFuncType,
    ProbeType,
)

# endregion

@dataclass
class ProbeFuncConfig:
    """Stores attributes for use with the ProbingFuncGen() Class. related to probing functions and their required modifiers."""
    table_capacity: int

    # pertubation probing
    perturb_step_modifier: int = 5
    peturb_shift: int = 5

    # quadratic probing
    linear_term: int = 1
    qudratic_term: int = 3

    # random probing
    knuth_multiplicative_constant = 2654435761
    bit_size = 2**32

    # second universal hash - for use with universal double hashing. (its the step size hash)
    uni_second_scale: int = field(init=False)
    uni_second_shift: int = field(init=False)
    uni_second_prime: int = field(init=False)

    def __post_init__(self):
        """needed for computed attributes"""
        self.recompute(self.table_capacity)

    @staticmethod
    def _is_prime_number(number: int):
        """Boolean Check if number is a prime."""
        if number < 2:
            return False
        for i in range(2, int(math.isqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    @staticmethod
    def find_next_prime_number(table_capacity: int):
        """Finds the next prime number larger than the current table capacity."""
        candidate = table_capacity + 1
        while True:
            if ProbeFuncConfig._is_prime_number(candidate):
                return candidate
            candidate += 1

    def recompute(self, new_capacity):
        """recomputes the table capacity"""
        self.table_capacity = new_capacity
        # pick a prime larger than table_capacity (e.g., next prime > capacity * 1000)
        self.uni_second_prime = ProbeFuncConfig.find_next_prime_number(self.table_capacity * 1000)
        self.uni_second_scale = random.randint(1, self.uni_second_prime - 1)
        self.uni_second_shift = random.randint(0, self.uni_second_prime - 1)


class ProbeFuncGen:
    """Selects from a series of probing functions for use in Open Addressing Hash Tables."""
    def __init__(self, config: ProbeFuncConfig, second_hash_code: HashCode, start_index: Index, probe_count: int,) -> None:
        self._start_index = start_index
        self._probe_count = probe_count
        # should be generated from the same key. 
        self._second_hash_code = second_hash_code
        # composed objects
        self._config = config

    def select_probing_function(self, probe: ProbeType) -> Index:
        """choose which probing function to use"""
        if probe == ProbeType.LINEAR:
            return ProbeFuncLib.linear_probing_function(self._start_index, self._probe_count, self._config.table_capacity)
        elif probe == ProbeType.QUADRATIC:
            return ProbeFuncLib.quadratic_probing_function(self._start_index, self._config.linear_term, self._config.qudratic_term, self._probe_count, self._config.table_capacity) 
        elif probe == ProbeType.DOUBLE_HASH:
            step_size_index = ProbeFuncLib.doublehash_stepsize_compress_func(self._second_hash_code, self._config.table_capacity)
            return ProbeFuncLib.double_hashing(self._start_index, step_size_index, self._probe_count, self._config.table_capacity)
        elif probe == ProbeType.DOUBLE_UNIVERSAL:
            step_size_index = ProbeFuncLib.universal_step_hash_func(self._second_hash_code, self._config.uni_second_scale, self._config.uni_second_shift, self._config.uni_second_prime, self._config.table_capacity)
            return ProbeFuncLib. double_hashing(self._start_index, step_size_index, self._probe_count, self._config.table_capacity)
        elif probe == ProbeType.PERTURBATION:
            return ProbeFuncLib.pertubation_probing(self._start_index, self._config.perturb_step_modifier, self._config.peturb_shift, self._probe_count, self._config.table_capacity)
        elif probe == ProbeType.RANDOM:
            return ProbeFuncLib.random_probing(self._second_hash_code, self._probe_count, self._config.knuth_multiplicative_constant, self._config.bit_size, self._config.table_capacity)
        else:
            raise KeyInvalidError("Error: Invalid Enum Type Entered. Enter a valid enum type.")

class ProbeFuncLib:
    """A collection of probe functions for Open Addressing Hash Tables"""
    # ----- Compress Function -----
    @staticmethod
    def universal_step_hash_func(hash_code: HashCode, scale: int, shift: int, prime: int, table_capacity: int):
        """universal step size hash function for use with universal double hashing"""
        return 1 + ((scale * hash_code + shift) % prime) % (table_capacity -1)

    @staticmethod
    def doublehash_stepsize_compress_func(hash_code: HashCode, table_capacity: int) -> int:
        """creates a simple second hash function for step size for double hashing"""
        return 1 + (hash_code % (table_capacity - 1))
    
    # ----- Probing Function -----
    @staticmethod
    def linear_probing_function(start_index: Index, probe_count, table_capacity) -> Index:
        """traverses through hashtable looking for empty slot"""
        return (start_index + probe_count) % table_capacity

    @staticmethod
    def quadratic_probing_function(start_index: Index, linear_term: int, quadratic_term: int, probe_count: int, table_capacity: int) -> Index:
        """quadratic probing function."""
        linear_term = linear_term  # linear term - stops quad from missing slots
        quadratic_term = quadratic_term  # quadratic term - provides spread to probes
        return (start_index + linear_term * probe_count + quadratic_term * (probe_count**2)) % table_capacity

    @staticmethod
    def double_hashing(start_index: Index, step_size_index: Index, probe_count: int, table_capacity: int) -> Index:
        """Double Hashing - uses second hash as a step size - better spread probing function"""
        return (start_index + probe_count * step_size_index) % table_capacity

    @staticmethod
    def pertubation_probing(start_index: Index, step_modifier: int, pertub_bitshift: int, probe_count: int, table_capacity: int) -> Index:
        """modifies the original hashcode via bitshifting and uses it as a step size for probing."""
        perturb = start_index
        new_index = (start_index * step_modifier + 1 + perturb + probe_count) % table_capacity
        perturb >>= pertub_bitshift  # bitshift
        return new_index

    @staticmethod
    def random_probing(hash_code: HashCode, probe_count: int, knuth_constant: int, bit_size: int, table_capacity: int) -> Index:
        """Uses a random sequence to select the next index"""
        knuth_multiplicative_constant = knuth_constant
        bit_size = bit_size
        # this works as a step size.
        seed = (hash_code * knuth_multiplicative_constant) % bit_size
        step_size = seed % (table_capacity - 1) + 1 # ensures 1 <= step_size < table_capacity
        # random number seed is altered by the probe count. this number is deterministic.
        index = (hash_code % table_capacity + probe_count * step_size) % table_capacity
        return index
