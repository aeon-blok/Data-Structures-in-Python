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
    Literal,
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
from user_defined_types.generic_types import T, K, iKey
from user_defined_types.hashtable_types import LoadFactor, ValidateLoadFactor
from utils.constants import COLLISIONS_THRESHOLD
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray
from ds.maps.hash_functions import *
from ds.maps.probing_functions import *

# endregion


class MapUtils:
    """A collection of Utilities for Map Data Structures (hash tables, sets etc)"""
    def __init__(self, map_obj) -> None:
        self.obj = map_obj
        self._ansi = Ansi()

    # -------------------------------- Utilities   --------------------------------
    def _is_prime_number(self, number):
        """Boolean Check if number is a prime."""
        if number < 2:
            return False
        for i in range(2, int(math.isqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    def find_next_prime_number(self, table_capacity):
        """Finds the next prime number larger than the current table capacity."""
        candidate = table_capacity + 1
        while True:
            if self._is_prime_number(candidate):
                return candidate
            candidate += 1

    def check_key_type(self, key):
        """Checks the input key type with the stored hash table key type."""
        if self.obj._keytype is None:
            self.obj._keytype = type(key)
        elif type(key) != self.obj._keytype:
            raise KeyInvalidError(f"Error: Input Key Type Invalid. Expected: {self.obj._keytype.__name__}, Got: {key.datatype.__name__}")

    # -------------------------------- Chaining Hash Table Visualization  --------------------------------
    def _load_factor_color_indicator(self):
        """changes the color of the load factor text depending on a threshold"""
        # load factor - with color that changes to warn end user.
        if self.obj.current_load_factor < 0.65:
            load_factor_number = self._ansi.color(f"{self.obj.current_load_factor:.2f}", self._ansi.GREEN)
        else:
            load_factor_number = self._ansi.color(f"{self.obj.current_load_factor:.2f}", self._ansi.RED)
        load_factor_string = f"Load Factor: {load_factor_number}"
        return load_factor_string

    def _collisions_color_indicator(self):
        """changes the color of the collisions text depending on a threshold"""
        # total collisions - with color change
        collision_threshold: float = COLLISIONS_THRESHOLD  # percentage boundary (13%)
        if self.obj.total_collisions / self.obj.table_capacity < collision_threshold:
            collisions_number = self._ansi.color(f"{self.obj.total_collisions}", self._ansi.GREEN)
        else:
            collisions_number = self._ansi.color(f"{self.obj.total_collisions}", self._ansi.RED)
        total_coll_string = f"Total Collisions: {collisions_number}"
        return total_coll_string

    def _chaining_hash_table_title(self, row_sep, load_factor_string, collisions_string):
        """creates a title with important stats for the viewer"""
        print(row_sep)
        hashtable_type_string = self._ansi.color(f"(Type: [{self.obj.datatype.__name__}])", self._ansi.BLUE)
        rehash_stats = f"Total Rehashes: {self.obj.total_rehashes}, Rehash Time (total): {self.obj.total_rehash_time:.1f} secs"
        stats = f"{load_factor_string}, {collisions_string}, Current Capacity: {self.obj.total_elements}/{self.obj.table_capacity}, Total Buckets Created: {self.obj.total_buckets}"
        title = self._ansi.color(f"Hash Table Visualization ", self._ansi.YELLOW) + hashtable_type_string
        print(title.center(len(row_sep)))
        print(row_sep)
        print(stats.center(len(row_sep)))
        print(rehash_stats.center(len(row_sep)))
        print(row_sep)

    def _populate_table(self):
        """searches the table, Counts the total number of kv pairs in each bucket & generates strings to be used in visualizing the table."""
        table = self.obj.buckets.array
        table_container = []
        # loops through every bucket in the table. appends the index number and count of keys for each bucket with items.
        # otherwise appends an empty list. (we will fill this in later with placeholder text.)
        for idx, bucket in enumerate(table):
            bucket_container = []
            if bucket is None:
                table_container.append([])
            if bucket is not None:
                count = len(bucket) if bucket else 0 # type: ignore
                stats = f"i:{idx} k:{count}"
                bucket_container.append(stats)  # append found items to the bucket container
                table_container.append(bucket_container)    # append buckets to the table container.
        return table_container

    def view_chaining_table(self, columns: int=16, cell_width:int = 11, row_padding: int = 3):
        """Visualizes the hash table as a console cell grid. contains the index number and number of keys in each bucket for clarity."""

        # table creation.
        columns = columns
        cell_width = cell_width
        row_seperator = "-" * (columns * (cell_width + row_padding))

        # title creation
        load_factor_string = self._load_factor_color_indicator()
        total_coll_string = self._collisions_color_indicator()
        self._chaining_hash_table_title(row_seperator, load_factor_string, total_coll_string)

        # create rows and populate
        populated_table = self._populate_table()
        table_size = len(populated_table)

        # add populated data to visualization.
        print(row_seperator)
        for i in range(0, table_size, columns):
            row = populated_table[i:i+columns]  # slices table container to create a sublist for each row of size columns.
            row_display = []
            # for every bucket in the sliced part of the table - if its empty append a placeholder, otherwise append the stats text
            for bucket in row:
                if not bucket:  # if the bucket is empty (the list representation of a bucket)
                    row_display.append("[]".center(cell_width))
                else:
                    row_display.append(", ".join(str(f"{stats}") for stats in bucket).center(cell_width))
            print(f"{' | '.join(row_display)}")
            print(row_seperator)


    # -------------------------------- Table Rehashing   --------------------------------

    def calculate_load_factor(self, total_elements: int, table_capacity: int) -> LoadFactor:
        """calculates the load factor of the current hashtable"""
        load = total_elements / table_capacity
        return ValidateLoadFactor(load)
