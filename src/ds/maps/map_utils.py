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
from user_defined_types.hashtable_types import LoadFactor, NormalizedFloat
from utils.constants import (
    COLLISIONS_THRESHOLD,
    TOMBSTONE_MARKER,
    LOAD_FACTOR_SYMBOL,
    COLLISIONS_SYMBOL,
    REHASH_SYMBOL,
    PROBE_SYMBOL,
    AVERAGE_PROBES_SYMBOL,
)

from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.helpers import Ansi

from user_defined_types.generic_types import Index
from user_defined_types.hashtable_types import ProbeType

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.map_adt import MapADT
    from adts.sequence_adt import SequenceADT

from adts.set_adt import SetADT
from ds.primitives.arrays.dynamic_array import VectorArray

# endregion


class MapUtils:
    """A collection of Utilities for Map Data Structures (hash tables, sets etc)"""
    def __init__(self, map_obj) -> None:
        self.obj = map_obj
        self._ansi = Ansi()

    # -------------------------------- Utilities --------------------------------
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
        if self.obj._table_keytype is None:
            self.obj._table_keytype = key.datatype
        elif key.datatype != self.obj._table_keytype:
            raise KeyInvalidError(f"Error: Input Key Type Invalid. Expected: {self.obj._table_keytype.__name__}, Got: {key.datatype.__name__}")

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

    def _populate_chain_table_view(self):
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
        populated_table = self._populate_chain_table_view()
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

    # -------------------------------- Table Rehashing --------------------------------
    def calculate_load_factor(self, total_elements: int, table_capacity: int) -> LoadFactor:
        """calculates the load factor of the current hashtable"""
        load = total_elements / table_capacity
        return NormalizedFloat(load)

    # -------------------------------- Open Addressing Hash Table  --------------------------------

    def rehash_condition(self) -> bool:
        """will rehash the table if any 1 of these conditions is true."""
        if self.obj.current_load_factor > self.obj.max_load_factor:
            return True
        elif self.obj.probe_ratio > self.obj.probe_threshold:
            return True
        elif self.obj.tombstones_ratio > self.obj.tombstones_threshold:
            return True
        elif self.obj.average_probe_length > self.obj.average_probe_limit:
            return True
        return False

    # -------------------------------- Visualizing Open Addressing Hash Table  --------------------------------

    def load_factor_stats_OA_indicator(self, color: bool = True):
        """changes the color of the load factor text depending on a threshold -- and provides a symbol for easy identification"""
        # Load Factor:
        load_factor_emoji = LOAD_FACTOR_SYMBOL
        if self.obj.current_load_factor < self.obj.max_load_factor:
            color_load_factor = self._ansi.color(f"{self.obj.current_load_factor:.2f}", self._ansi.GREEN)
        else: 
            color_load_factor = self._ansi.color(f"{self.obj.current_load_factor:.2f}", self._ansi.RED)
        load_factor_string = f"{load_factor_emoji} : {color_load_factor}"
        load_factor_nocolor = f"{load_factor_emoji} : {self.obj.current_load_factor:.2f}"
        return load_factor_string if color else load_factor_nocolor

    def collisions_stats_OA_indicator(self, color: bool = True):
        """changes the color of the collisions text depending on a threshold -- and provides a symbol for easy identification"""
        collisions_emojis = COLLISIONS_SYMBOL
        if self.obj.collisions_ratio < self.obj.collisions_threshold - 0.05:
            color_collisions = self._ansi.color(f"{self.obj.current_collisions}", self._ansi.GREEN)  
        else: 
            color_collisions = self._ansi.color(f"{self.obj.current_collisions}", self._ansi.RED)
        total_collisions_string = f"{collisions_emojis} : {color_collisions}"
        total_coll_nocolor = f"{collisions_emojis} : {self.obj.current_collisions}"
        return total_collisions_string if color else total_coll_nocolor

    def tombstone_stats_OA_indicator(self, color: bool = True):
        """changes the color of the tombstone stats depending on a threshold -- and provides a symbol for easy identification"""
        tombstone_emojis = TOMBSTONE_MARKER
        if  self.obj.tombstones_ratio < self.obj.tombstones_threshold - 0.05:
            color_tombstones = self._ansi.color(f"{self.obj.current_tombstones}",self._ansi.GREEN)  
        else: 
            color_tombstones = self._ansi.color(f"{self.obj.current_tombstones}", self._ansi.RED)
        tombstone_string = f"{tombstone_emojis}  : {color_tombstones}"
        tombstones_nocolor = f"{tombstone_emojis}  : {self.obj.current_tombstones}"
        return tombstone_string if color else tombstones_nocolor

    def rehash_stats_OA_indicator(self):
        """rehash indicator with symbol"""
        rehash_emoji = REHASH_SYMBOL
        rehashes_string = f"{rehash_emoji}  : {self.obj.total_rehashes}"
        return rehashes_string

    def probe_stats_OA_indicator(self, color: bool = True):
        """probe stats with symbol -- indicates the current probe length. (amount of slots traversed till an empty slot is found.)"""
        probe_emoji = PROBE_SYMBOL
        if self.obj.probe_ratio < self.obj.probe_threshold - 0.05:
            color_probes = self._ansi.color(f"{self.obj.current_probes}", self._ansi.GREEN)
        else: 
            color_probes = self._ansi.color(f"{self.obj.current_probes}", self._ansi.RED)
        probes_string = f"{probe_emoji} : {color_probes}"
        probes_nocolor = f"{probe_emoji} : {self.obj.current_probes}"
        return probes_string if color else probes_nocolor

    def average_probe_length_stats_OA_indicator(self, color: bool = True):
        """shows the average probe number (as a float) -- with a symbol and color indicator for danger levels."""
        average_probe_emoji = AVERAGE_PROBES_SYMBOL
        if self.obj.average_probe_length < 3:
            color_avg_probe = self._ansi.color(f"{self.obj.average_probe_length:.2f}" , self._ansi.GREEN)  
        else: 
            color_avg_probe = self._ansi.color(f"{self.obj.average_probe_length:.2f}", self._ansi.RED)
        avg_probes_string = f"{average_probe_emoji} : {color_avg_probe}"
        avg_probes_nocolor = f"{average_probe_emoji} : {self.obj.average_probe_length:.2f}"
        return avg_probes_string if color else avg_probes_nocolor

    def _populate_OA_hash_table_view(self):
        """
        creates a list of entries for all spaces in the table.
        empty spaces are blank
        tombstones - have a unique marker.
        occupied slots - have the index number.
        """
        table = self.obj.table.array
        table_container = []
        # traverse every item in table
        # - if there is an item add the index number as text to the slot. - otherwise add the tombstone marker or []
        for idx, item in enumerate(table):
            if item == self.obj.tombstone:
                table_container.append(TOMBSTONE_MARKER)
            elif item is None:
                table_container.append("")
            else:
                table_container.append(f"i: {idx}")
        return table_container

    def create_OA_hash_table_title(self, row_seperator):
        """Creates a title for the Open Addressing Hash Table Console Visualization"""
        # title
        print(row_seperator)
        hashtable_type_string = f"(Type: [{self.obj.enforce_type.__name__}])"
        title = f"Hash Table Visualization: {hashtable_type_string}"
        stats = f"{self.obj.datatype_string}{self.obj.capacity_string}[{self.obj.loadfactor_string}, {self.obj.probes_string}, {self.obj.tombstone_string}, {self.obj.total_collisions_string}, {self.obj.rehashes_string}, {self.obj.avg_probes_string}]"
        print(title.center(len(row_seperator)))
        print(row_seperator)
        print(stats.center(len(row_seperator)))
        print(row_seperator)

    def view_OA_hash_table(self, columns: int = 12, cell_width: int = 15, row_padding: int = 3):
        """a console visualization of an Open Addressing Hash Table"""

        # table creation.
        columns = columns
        cell_width = cell_width
        row_seperator = "-" * (columns * (cell_width + row_padding))
        # populate with entries for display.
        populated_table = self._populate_OA_hash_table_view()

        # create title
        self.create_OA_hash_table_title(row_seperator)

        # create rows & rows logic
        table_size = len(populated_table)
        for i in range(0, len(populated_table), columns):
            row = populated_table[i : i + columns]
            row_display = [str(item).center(cell_width) for item in row]
            print(" | ".join(row_display))
            print(row_seperator)

    # region Hash Set
    # -------------------------------- Hash Set  --------------------------------

    def validate_set(self, set):
        """checks to see that input is a valid set and not a none value"""
        if set is None:
            raise DsTypeError("Error: Set Cannot be a none value")
        if not isinstance(set, SetADT):
            raise DsTypeError(f"Error: Input is not a Set. Must match and implement SetADT interface. Got: {type(set).__name__}")

    def check_set_empty(self):
        if self.obj.is_empty:
            raise DsUnderflowError(f"Error: Set is Empty...")

    # endregion

    # region Skip List
    # -------------------------------- Skip List (Sorted Map)  --------------------------------

    def set_skiplist_keytype(self, key):
        """On first insertion - this will set the keytype of the skip list to be the same type as the inserted key."""
        if self.obj.keytype is None:
            self.obj._keytype = key.datatype

    def check_ketype_is_same(self, key):
        """ensures the keys are comparable"""
        if key.datatype != self.obj.keytype:
            raise KeyInvalidError(f"Error: Input Key Type Invalid. Expected: {self.obj.keytype.__name__}, Got: {key.datatype.__name__}")


    # endregion
