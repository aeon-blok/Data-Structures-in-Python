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

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT


# endregion



class ProbingFunctions:
    """A collection of probe functions for Open Addressing Hash Tables"""
    def __init__(self, map_obj) -> None:
        self.obj = map_obj

    # ----- Probing Function -----
    def linear_probing_function(self, index) -> int:
        """traverses through hashtable looking for empty slot"""
        return (index + 1) % self.capacity

    def quadratic_probing_function(self, start_index, probe_count) -> int:
        """quadratic probing function."""
        linear_term = 1  # linear term - stops quad from missing slots
        quadratic_term = 3  # quadratic term - provides spread to probes
        return (start_index + linear_term * probe_count + quadratic_term * (probe_count**2)) % self.capacity

    def double_hashing(self, key, start_index, probe_count) -> int:
        """Double Hashing - uses second hash as a step size - better spread probing function"""
        second_step_size_index = self._second_hash_function(key)
        return (start_index + probe_count * second_step_size_index) % self.capacity

    def pertubation_probing(self, index) -> int:
        """modifies the original hashcode via bitshifting and uses it as a step size for probing."""
        perturb = index
        new_index = (index * 5 + 1 + perturb) % self.capacity
        perturb >>= 5  # bitshift
        return new_index

    def select_probing_function(self, key, index, start_index, probe_count) -> int:
        """Selects between different probing functions (quadratic, linear, double hashing)"""
        if self.probing_technique == "linear":
            new_index = self.linear_probing_function(index)
        elif self.probing_technique == "quadratic":
            new_index = self.quadratic_probing_function(start_index, probe_count)
        elif self.probing_technique == "double hashing":
            new_index = self.double_hashing(key, start_index, probe_count)
        elif self.probing_technique == "pertubation":
            new_index = self.pertubation_probing(index)
        else:
            raise ValueError(f"Error: {self.probing_technique}: Invalid Probing Technique entered. Please select from valid options.")
        return new_index