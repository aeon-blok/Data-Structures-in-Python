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
    NewType,
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

from utils.constants import (
    DEFAULT_HASHTABLE_CAPACITY,
    MAX_LOAD_FACTOR,
    PROBES_THRESHOLD,
    AVERAGE_PROBES_LIMIT,
    MIN_HASHTABLE_CAPACITY,
    TOMBSTONES_THRESHOLD,
    HASHTABLE_RESIZE_FACTOR,
    COLLISIONS_THRESHOLD,
)

from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.representations import OAHashTableRepr
from utils.helpers import RandomClass

from user_defined_types.generic_types import (
    T,
    K,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    Index,
)

from user_defined_types.hashtable_types import (
    ProbeType,
    HashCodeType,
    CompressFuncType,
    NormalizedFloat,
    LoadFactor,
    BitMask,
    Tombstone,
    PercentageFloat,
)

from user_defined_types.key_types import Key, iKey
from user_defined_types.hashtable_types import (
    HashCodeType,
    ProbeType,
    CompressFuncType,
    HashCode,
    NormalizedFloat,
    PercentageFloat,
    LoadFactor,
    BitMask,
)

from adts.map_adt import MapADT
from adts.collection_adt import CollectionADT

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.maps.map_utils import MapUtils
from ds.maps.probing_functions import ProbeFuncConfig, ProbeFuncGen
from ds.maps.hash_functions import HashFuncConfig, HashFuncGen

if TYPE_CHECKING:
    pass


# endregion

"""
A Hash Table is a data structure that stores key-value pairs via a hash function which computes an index into an underlying bucket array.
For this implementation we will handle collisions via Open Addressing & Linear Probing (via Tombstones)
"""


class HashTableOA(MapADT[T, K], CollectionADT[T], Generic[T, K]):
    """
    Hash Table Data Structure with Probing / double hashing & Tombstones (Open Addressing)
    self.return_keys: The Hash table has a property that allows it to return key() objects for easy comparison. (sorted, max, min etc)
    """
    def __init__(
        self,
        datatype: type,
        capacity: int = DEFAULT_HASHTABLE_CAPACITY,
        max_load_factor: LoadFactor = NormalizedFloat(MAX_LOAD_FACTOR),
        resize_factor: int = HASHTABLE_RESIZE_FACTOR,
        probes_threshold: PercentageFloat = NormalizedFloat(PROBES_THRESHOLD),
        tombstones_threshold: PercentageFloat = NormalizedFloat(TOMBSTONES_THRESHOLD),
        average_probes_limit: float = AVERAGE_PROBES_LIMIT,
        probing_technique: ProbeType = ProbeType.DOUBLE_UNIVERSAL,
        hash_code: HashCodeType = HashCodeType.CYCLIC_SHIFT,
        compress_func: CompressFuncType = CompressFuncType.MAD,
    ):

        # composed objects
        self._utils: MapUtils = MapUtils(self)
        self._validators: DsValidation = DsValidation()
        self._desc: OAHashTableRepr = OAHashTableRepr(self)

        # table size
        self.min_capacity: int = max(MIN_HASHTABLE_CAPACITY, self._utils.find_next_prime_number(capacity))
        self.table_capacity = self._utils.find_next_prime_number(capacity)

        # type safety
        self.enforce_type = ValidDatatype(datatype)
        # its the key specified type for the entire table
        self._table_keytype: type | None = None # the first key to be entered defines the type.

        # initialize table.
        self.table: VectorArray = VectorArray(self.table_capacity, object)
        for i in range(self.table_capacity):
            self.table.array[i] = None

        # core attributes
        self.total_elements = 0   # tracks the number of kv pairs in the table
        self.max_load_factor = max_load_factor # prevents the table from exceeding this capacity
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)  # log attribute - displays current load factor
        self.resize_factor = resize_factor

        # unique tombstone class. used as a tombstone marker
        self.tombstone = Tombstone()

        # Hashing: Composed Objects
        self._hash_code = hash_code
        self._compress_func = compress_func
        self._probing_technique = probing_technique
        # have to recompute the configs every rehash
        self._hashconfig: HashFuncConfig = HashFuncConfig(self.table_capacity)
        self._probeconfig: ProbeFuncConfig = ProbeFuncConfig(self.table_capacity)

        # region trackers
        self.current_collisions = 0
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_tombstones = 0  
        self.current_probes = 0
        self.total_probes = 0
        self.total_probe_operations = 0
        self.collisions_threshold: PercentageFloat = NormalizedFloat(COLLISIONS_THRESHOLD)
        self.tombstones_threshold = tombstones_threshold
        self.probe_threshold = probes_threshold
        self._collisions_ratio: PercentageFloat = self.current_collisions / self.table_capacity
        self._tombstones_ratio: PercentageFloat = self.current_tombstones / self.table_capacity
        self._probe_ratio: PercentageFloat = self.current_probes / self.table_capacity
        self._average_probe_length: float = 0.0
        self.average_probe_limit: float = average_probes_limit
        # endregion

    # region ratios
    @property
    def collisions_ratio(self) -> PercentageFloat:
        return self.current_collisions / self.table_capacity

    @property
    def tombstones_ratio(self) -> PercentageFloat:
        return self.current_tombstones / self.table_capacity

    @property
    def probe_ratio(self) -> PercentageFloat:
        return self.current_probes / self.table_capacity

    @property
    def average_probe_length(self) -> PercentageFloat:
        if self.total_elements == 0:
            return 0.0
        return self.total_probes / self.total_elements

    @average_probe_length.setter
    def average_probe_length(self, value):
        self._average_probe_length = value
    # endregion

    # region stats:
    @property
    def table_items(self) -> str:
        return f"{str(', '.join(f'{k}: {v}' for k, v in self.items()))}"

    @property
    def capacity_string(self) -> str:
        return f"[{self.total_elements}/{self.table_capacity}]"

    @property
    def datatype_string(self) -> str:
        return f"[{self.enforce_type.__qualname__}]"

    @property
    def loadfactor_string(self) -> str:
        return self._utils.load_factor_stats_OA_indicator()

    @property
    def probes_string(self) -> str:
        return self._utils.probe_stats_OA_indicator()

    @property
    def avg_probes_string(self) -> str:
        return self._utils.average_probe_length_stats_OA_indicator()

    @property
    def tombstone_string(self) -> str:
        return self._utils.tombstone_stats_OA_indicator()

    @property
    def total_collisions_string(self) -> str:
        return self._utils.collisions_stats_OA_indicator()

    @property
    def rehashes_string(self) -> str:
        return self._utils.rehash_stats_OA_indicator()

    # endregion

    @property
    def return_keys(self) -> VectorArray[Key]:
        """returns key objects.... Good for sorting & comparisons."""
        # Init Vector Array
        if self._table_keytype is None:
            found = VectorArray(self.table_capacity, object)
        else:
            found = VectorArray(self.table_capacity, iKey)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                found.append(k)
        return found
    
    # ----- Utility -----
    def _display_table(self, columns: int = 12, cell_width: int = 15, row_padding: int = 3):
        """Table visualization - with tombstone markers included!"""
        return self._utils.view_OA_hash_table(columns, cell_width, row_padding)

    # ----- Python Built in Overrides -----
    def __str__(self) -> str:
        """prints whenever the item is printed in the console"""
        return self._desc.str_oa_hashtable()

    def __repr__(self) -> str:
        """prints dev info"""
        return self._desc.repr_oa_hashtable()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)

    def __delitem__(self, key):
        return self.remove(key)

    # ----- Table Rehashing -----
    def _rehash_table(self):
        """
        Rehashes table - copies items from an old table to a new table - and resets tracking counters
        Step 1: Store Old Hash Table (for later copy)
        Step 2: Create & Initialize New Table with New Capacity (usually x2)
        Step 3: Reset trackers
        Step 4: Copy keys from old table to the new table. (use an internal_put() method)
        Step 5: Update Rehash trackers & Calculate Load Factor
        """
        start_time = time.perf_counter()

        # Store Old hash table
        old_capacity = self.table_capacity
        old_table = self.table.array

        # Set new capacity (normally * 2)
        new_capacity = self._utils.find_next_prime_number(old_capacity * self.resize_factor)
        # recompute attributes for hash function and probe function
        self._hashconfig.recompute(new_capacity)
        self._probeconfig.recompute(new_capacity)

        # initialize new table.
        new_table = VectorArray(new_capacity, object)
        for i in range(new_capacity):
            new_table.array[i] = None
        # reinitialize table with new size.
        self.table = new_table  
        self.table_capacity = new_capacity

        # reset trackers
        self.total_elements = 0
        self.current_collisions = 0
        self.current_tombstones = 0
        self.current_probes = 0
        # update average probe metrics
        self.average_probe_length = 0.0
        self.total_probes = 0
        self.total_probe_operations = 0

        # copy keys from old table to new table
        for slot in old_table:
            if slot is not None and slot != self.tombstone:
                old_k, old_v = slot
                self._internal_put(old_k,old_v)

        end_time = time.perf_counter()

        rehash_time = end_time - start_time
        self.total_rehash_time += rehash_time   # updates lifetime tracker of rehash time.
        self.total_rehashes += 1    # update total rehashes
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)

    def _internal_put(self, key, value):
        """For use with the rehash functionality only -- does not use the rehash condition."""

        # validate inputs
        key = Key(key)
        self._utils.check_key_type(key)
        value = TypeSafeElement(value, self.enforce_type)

        # * generate hash
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()
        second_hash_code = hashgen.create_hash_code()  # outside probing loop

        # initialize variables for probing loop
        start_index = index # set start index for probe function
        tombstone_start_index = None
        probe_count = 0

        # * Probing Loop:
        while self.table.array[index] is not None:
            probe_count += 1    # adds to probe count on keys and tombstones...
            # tombstone logic
            if self.table.array[index] == self.tombstone:
                if tombstone_start_index is None:   # only cache the first tombstone index we find.
                    tombstone_start_index = index # cache index to use for insertion
            # keys only - Update value if key already exists.
            else:
                k, v = kv_pair = self.table.array[index] 
                if k == key:
                    self.table.array[index] = (key, value)    # update value
                    self.current_probes = probe_count
                    return
            # add to collisions if we collide with a live key only
            if self.table.array[index] is not None and self.table.array[index] is not self.tombstone:
                self.current_collisions += 1 

            # apply probe func
            probegen = ProbeFuncGen(self._probeconfig, second_hash_code, start_index, probe_count)
            # moves to the next index on the table - This is the core of linear probing.
            index = probegen.select_probing_function(self._probing_technique)

            # Exit Condition: if we get back to where we started with no empty slot - the table is full
            if self._probing_technique == ProbeType.RANDOM:
                if probe_count >= self.table_capacity:
                    raise DsOverflowError(f"Error: Hash table is full.")
            else:
                if index == start_index:
                    raise DsOverflowError(f"Error: Hash table is full.")

        # * Default Condition: Add kv pair to index
        target_index = tombstone_start_index if tombstone_start_index is not None else index
        # equivalence check: if we replace a tombstone - decrement tombstones counter.
        if self.table.array[target_index] == self.tombstone:
            self.current_tombstones -= 1
        self.table.array[target_index] = (key, value)
        self.total_elements += 1
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
        self.current_probes = probe_count
        # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
        self.total_probes += self.current_probes
        self.total_probe_operations += 1

    # ----- Canonical ADT Operations -----
    def put(self, key, value):
        """
        Insert a key value pair into the hash table: -- Probing Function will search for the next empty slot.
        Step 1: Rehash Condition: Check if over load factor and rehash table
        Step 2: Hash Function: Calculate Index via Hash Function
        Step 3: Probing Function: Traverse the table, ignoring empty slots and tombstones -- If the key is found update the key value pair.
        Step 4: Default Condition: Update the key value pair & increment size tracker
        """

        # * table rehash conditions - always has to be first so that the key and hash functions are correctly applied.
        if self._utils.rehash_condition():
            self._rehash_table()

        # validate inputs
        key = Key(key)
        self._utils.check_key_type(key)
        value = TypeSafeElement(value, self.enforce_type)

        # generate hash
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()
        second_hash_code = hashgen.create_hash_code()   # outside probing loop

        # initialize variables for probing loop
        start_index = index # set start index for probe function
        tombstone_start_index = None
        probe_count = 0  # number of probes until key is found or insertion succeeds
        # * Probing Function: travel through the table - ignoring None and tombstones. (only actual kv pairs)
        while self.table.array[index] is not None:
            probe_count += 1    # adds on keys and tombstones
            # logic for tombstone -- only cache the first tombstone index we find...
            if self.table.array[index] == self.tombstone:
                if tombstone_start_index is None: tombstone_start_index = index
            # logic for keys
            else:
                slot = self.table.array[index]
                k, v = slot  # type: ignore
                # Update value if key already exists
                if k == key:
                    self.table.array[index] = (key, value)  # update value
                    return

            # add to collisions if we collide with a live key only
            if self.table.array[index] is not None and self.table.array[index] is not self.tombstone:
                self.current_collisions += 1 

            # apply probe func
            probegen = ProbeFuncGen(self._probeconfig, second_hash_code, start_index, probe_count)
            # moves to the next index on the table - This is the core of linear probing.
            index = probegen.select_probing_function(self._probing_technique)

            # Error/Exit Condition: if we get back to where we started with no empty slot - the table is full
            if self._probing_technique == ProbeType.RANDOM:
                if probe_count >= self.table_capacity: 
                    raise DsOverflowError(f"Error: Hash table is full.")
            else:
                if index == start_index:
                    raise DsOverflowError(f"Error: Hash table is full.")

        # * Default Condition: Add kv pair to index
        # defines the index as either the first tombstone that was found, or the current index.
        target_index: int = tombstone_start_index if tombstone_start_index is not None else index
        # equivalence check: if we replace a tombstone - decrement tombstones counter.
        if self.table.array[target_index] == self.tombstone: 
            self.current_tombstones -= 1
        self.table.array[target_index] = (key, value)
        # updates trackers
        self.total_elements += 1
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
        self.current_probes = probe_count
        # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
        self.total_probes += self.current_probes
        self.total_probe_operations += 1    

    def get(self, key, default=None):
        """retrieves the element value from a kv pair from the hash table, with an optional default if the key is not found."""

        # validate inputs
        key = Key(key)
        self._utils.check_key_type(key)

        if default is not None:
            default = TypeSafeElement(default, self.enforce_type)

        # generate hash
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()
        second_hash_code = hashgen.create_hash_code()  # outside probing loop

        start_index = index
        probe_count = 0

        # traverse table - ignore empty slots (do NOT ignore tombstones during retrieval.)
        while self.table.array[index] is not None:
            probe_count += 1
            slot = self.table.array[index]
            if slot != self.tombstone:
                k, v = slot
                if k == key:
                    return v

            # apply probe func
            probegen = ProbeFuncGen(self._probeconfig, second_hash_code, start_index, probe_count)
            # moves to the next index on the table - This is the core of linear probing.
            index = probegen.select_probing_function(self._probing_technique)

            # Exit Condition: if we have traversed the whole table and nothing found, break while loop and return default.
            if self._probing_technique == ProbeType.RANDOM:
                if probe_count >= self.table_capacity: break
            else:
                if index == start_index: break

        self.current_probes = probe_count
        # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
        self.total_probes += self.current_probes
        self.total_probe_operations += 1

        return default

    def remove(self, key):
        """
        removes a key value pair from the hash table. deleting a key - involves lazy deletion(archiving) & tombstone markers.
        Step 1: Hash Function
        Step 2: Traverse Table - Check if slot has the valid key, (or if tombstone or null). Add tombstone marker and return value.
        Step 3: Raise error if specified key not found for deletion.
        """

        # rehash condition:
        if self._utils.rehash_condition():
            self._rehash_table()

        # validate inputs
        key = Key(key)
        self._utils.check_key_type(key)

        # generate hash
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()
        second_hash_code = hashgen.create_hash_code()  # outside probing loop

        start_index = index
        probe_count = 0

        # find key at index. (skip None and Tombstone markers)
        while True: 
            probe_count += 1
            slot = self.table.array[index]

            if slot is not None and slot != self.tombstone:
                k, v = slot
                # if the key matches - add tombstone marker to the table index specifically
                if k == key:
                    self.table.array[index] = self.tombstone    # the act of DELETION!
                    # update trackers.
                    self.total_elements -= 1
                    self.current_tombstones += 1
                    self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
                    # update current probes metric for trackers
                    self.current_probes = probe_count
                    # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
                    self.total_probes += self.current_probes
                    self.total_probe_operations += 1
                    return v

            # apply probe func
            probegen = ProbeFuncGen(self._probeconfig, second_hash_code, start_index, probe_count)
            # moves to the next index on the table - This is the core of linear probing.
            index = probegen.select_probing_function(self._probing_technique)

            # Exit Condition: looped the whole way round....
            if self._probing_technique == ProbeType.RANDOM:
                if probe_count >= self.table_capacity:
                    break
            else:
                if index == start_index:
                    break

        # raise error if no key found....
        raise KeyError(f"Error: Key: {key} not found...")

    def keys(self):
        """Return a set of all the keys in the hash table"""
        # Init Vector Array
        if self._table_keytype is None:
            found = VectorArray(self.table_capacity, object)
        else:
            found = VectorArray(self.table_capacity, self._table_keytype)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                k = k.value  # unpack key object.
                found.append(k)
        return found

    def values(self):
        """Return a set of all the values in the hash table"""
        found = VectorArray(self.table_capacity, self.enforce_type)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                found.append(v)
        return found

    def items(self):
        """Return a set of all the values in the hash table"""
        found = VectorArray(self.table_capacity, tuple)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                key = k.value  # unpack key object.
                value = v
                kv_pair = (key, value)  # pack again with unpacked key value
                found.append(kv_pair)
        return found

    # ----- Meta Collection ADT Operations -----
    def __len__(self):
        """Returns the number of key-value pairs in the hash table"""
        return self.total_elements

    def __contains__(self, key):
        """Does the Hash table contain an item with the specified key?"""
        # validate inputs
        key = Key(key)
        self._utils.check_key_type(key)

        # generate hash
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        # initialize start index and probe count for probing loop.
        start_index = index
        probe_count = 0

        # might break - change to while true etc
        while self.table.array[index] is not None:
            probe_count += 1
            if self.table.array[index] != self.tombstone:
                k, v = self.table.array[index]
                if k == key: return True
            # apply probe func
            second_hash_code = hashgen.create_hash_code()
            probegen = ProbeFuncGen(self._probeconfig, second_hash_code, start_index, probe_count)
            # moves to the next index on the table - This is the core of linear probing.
            index = probegen.select_probing_function(self._probing_technique)
            if index == start_index: break  # exit condition
        return False

    def is_empty(self):
        return self.total_elements == 0

    def clear(self):
        """Resets the table to initial empty space with original capacity. resets all trackers also."""
        # reset trackers
        self.table_capacity = self.min_capacity
        # recompute attributes for hash function.
        self._hashconfig.recompute(self.table_capacity)
        self._probeconfig.recompute(self.table_capacity)

        self.total_elements = 0  # reset item count
        self.current_tombstones = 0 # reset tombstones count
        self.current_collisions = 0   # reset collisions count
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_probes = 0
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)    # reset load factor
        # update average probe metrics
        self.average_probe_length = 0.0
        self.total_probes = 0
        self.total_probe_operations = 0

        # reinitialize table.
        self.table = VectorArray(self.table_capacity, object)   
        for i in range(self.table_capacity):
            self.table.array[i] = None

    def __iter__(self):
        """The default iteration for a Map, is to generate a sequence (list) of all the keys in the map."""
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                k = k.value # unpack key object.
                yield k


# todo keep on the lookout for the following flaky bugs. (probably solved - tested with 25,000 entries -- hundreds of times.)
# ! in put raise DsOverflowError(f"Error: Hash table is full.")
# ! this happened dude to a bug in random probing,
# !(if the random stepsize was coprime with the knuth constant, it would make an index unreachable.)

# ! Key 59 not found Error: this occurs in remove()
# ! seems to happen when close to rehashing.
# ! a potential interplay between the hashcode and the probing loop
# !- moving the 2nd hashcode generated out of the probing loop seems to have fixed it for now.
# todo write proper tests. cover all functionality.

# Main ---- Client Facing Code ------
def main():

    # region input data
    # ---- inpput data -----

    AI = type(
        "ArtificialPerson",
        (),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"NotAPerson({self.name})",
            "__repr__": lambda self: f"NotAPerson({self.name})",
        },
    )

    wrong_type = AI("bob")

    # 1. Integers
    int_list = [random.randint(-1000, 1000) for _ in range(20)]

    # 2. Floats
    float_list = [random.uniform(-1000.0, 1000.0) for _ in range(20)]

    # 3. Booleans
    bool_list = [random.choice([True, False]) for _ in range(20)]

    # Dictionaries
    preset_dicts = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
        {"color": "red", "value": "#FF0000"},
        {"color": "green", "value": "#00FF00"},
    ]

    # Lists
    preset_lists = [[1, 2, 3], ["a", "b", "c"], [True, False, True], [42, 43, 44], []]

    # Tuples
    preset_tuples = [(1, 2, 3), ("x", "y", "z"), (True, False, True), (42, 43, 44), ()]

    # Preset names for Person objects
    Person = type(
        "Person",
        (),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"Person({self.name})",
            "__repr__": lambda self: f"Person({self.name})",
        },
    )
    person_names = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Eve",
        "Frank",
        "Grace",
        "Hank",
        "Ivy",
        "Jack",
    ]
    preset_dynamic_objects = [Person(name) for name in person_names]

    string_data = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince",
        "raspberry",
        "strawberry",
        "tangerine",
        "ugli",
        "watermelon",
    ]

    input_data = preset_dynamic_objects

    input_values = [*string_data * 10]
    random.shuffle(input_values)

    # endregion

    # ht = HashTableOA(str)
    # for i in string_data:
    #     key = str(i)
    #     ht.put(key, i)
    #     print(f"get operation: {ht.get(key)}")
    #     print(repr(ht))
    # print(ht)

    # -- Initialize Hash Table ---
    hashtable = HashTableOA(str, capacity=20, max_load_factor=0.6, probing_technique=ProbeType.DOUBLE_HASH)
    print("Created hash table:", hashtable)

    # testing put() logic
    print(f"\nTesting Insertion Logic:")
    for i, key in enumerate(input_values):
        hashtable.put(f"key_{i}", key)
        print(repr(hashtable))    # testing __str__

    # testing remove logic
    print(f"\nTesting remove logic:")
    delete_items = list(hashtable.items())
    delete_subset = random.sample(delete_items, min(len(delete_items) // 2, 10000))
    for pair in delete_subset:
        k,v = pair
        hashtable.remove(k)
        print(repr(hashtable))

    # testing __getitem & __setitem__
    items = list(hashtable.items())
    keys = [k for k in hashtable.keys() if k is not None]
    subset = random.sample(items, min(5, 10))
    for i, kv_pair in enumerate(subset):
        random_key = random.choice(keys)
        random_key_value = hashtable[random_key]
        k,v = kv_pair
        getitem = hashtable[random_key]
        print(f"Retrieving Item [{random_key}] from Table: Got: [{getitem}]")
    for k, v in subset:
        random_value = random.choice(hashtable.values())
        print(f"Updating Value: {hashtable[k]} with new value {random_value}...")
        hashtable[k] = random_value
        print(f"Expected: {random_value} Got: {hashtable[k]}")

    print(f"\nSorting Keys and playing around....")
    keys = hashtable.return_keys
    sorted_keys = sorted(keys)
    # print(sorted_keys)
    print(f"getting max key. {max(keys)}")
    print(f"getting min key. {min(keys)}")

    # test type safety:
    try:
        print(f"\nTesting Invalid type input: {wrong_type}")
        hashtable.put("wrong_type", wrong_type)
    except Exception as e:
        print(f"{e}")

    # test __contains__
    print(f"\nCheck if Invalid Type: {wrong_type}: Exists in the table currently?\nExpected: False, Got: {hashtable.__contains__('wrong_type')}\n")

    # testing put() logic -- reinserting to test out tombstones.....
    new_items = list(hashtable.items())
    random.shuffle(new_items)
    subset = random.sample(new_items, min(80, len(items) // 4))
    for i, pair in enumerate(subset):
        k,v = pair
        hashtable.put(f"newkey_{k}_{i}", v)
        print(repr(hashtable))

    # test __len__
    print(f"Total Elements in Hash Table Currently: {len(hashtable)}")

    # display table
    hashtable._display_table()
    print(repr(hashtable))

    # print(f"\nTesting Keys(), Values(), items() etc...")
    # keys = hashtable.keys()
    # items = hashtable.items()
    # values = hashtable.values()
    # print(keys)

    # test clear()
    print(f"Clearing Table: ")
    hashtable.clear()
    print(repr(hashtable))
    print(f"Total Elements in Hash Table Currently: {len(hashtable)}")


if __name__ == "__main__":
    main()
