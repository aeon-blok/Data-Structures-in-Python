# region standard lib
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
)

from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import secrets
import math
import random
import time
import uuid
from pprint import pprint

# endregion

# region custom imports
from user_defined_types.generic_types import T, K, ValidDatatype, ValidIndex, TypeSafeElement, Index
from user_defined_types.hashtable_types import NormalizedFloat, LoadFactor, HashCodeType, CompressFuncType
from user_defined_types.key_types import iKey, Key

from utils.constants import MIN_HASHTABLE_CAPACITY, BUCKET_CAPACITY, HASHTABLE_RESIZE_FACTOR, DEFAULT_HASHTABLE_CAPACITY, MAX_LOAD_FACTOR

from utils.validation_utils import DsValidation
from utils.representations import ChainHashTableRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.map_adt import MapADT

from ds.maps.map_utils import MapUtils
from ds.maps.hash_functions import HashFuncConfig, HashFuncGen
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView


# endregion


class ChainHashTable(MapADT[T, K], CollectionADT[T], Generic[T, K]):
    """
    Hash Table implementation with collision chaining via bucket arrays.
    Essentially we build a Multi Dimensional Array. (MD Array) - the first array stores bucket arrays. the bucket arrays store kv pairs
    uses key() objects internally for key hashing and comparison. this standardizes comparisons. Can return the key() objects with property self.return_keys
    """
    def __init__(
        self,
        datatype: type,
        table_capacity: int = DEFAULT_HASHTABLE_CAPACITY,
        hash_code: HashCodeType = HashCodeType.SHA256,
        compress_func: CompressFuncType = CompressFuncType.SHA256,
        max_load_factor: LoadFactor = NormalizedFloat(MAX_LOAD_FACTOR),
        resize_factor: int = HASHTABLE_RESIZE_FACTOR,
    ) -> None:

        # composed objects:
        self._utils: MapUtils = MapUtils(self)
        self._validators: DsValidation = DsValidation()
        self._desc: ChainHashTableRepr = ChainHashTableRepr(self)

        # homogenous type safety
        self.datatype = ValidDatatype(datatype)
        # its the key specified type for the entire table. the first key to be entered defines the type.
        self._table_keytype: type | None = None   

        # trackers
        self.total_elements: int = 0   # number of key-value pairs
        self.total_buckets: int = 0  # number of created buckets.
        self.total_collisions: int = 0 # tracks the number of collisions that have occured
        self.total_rehashes: int = 0
        self.total_rehash_time: float = 0.0
        self.current_rehash_time: float = 0.0

        # core attributes
        self.resize_factor = resize_factor
        self.table_capacity = max(MIN_HASHTABLE_CAPACITY, self._utils.find_next_prime_number(table_capacity))    # number of slots in hash table
        self.bucket_capacity: int = self._utils.find_next_prime_number(BUCKET_CAPACITY) # initializes each bucket with this number of slots.
        self.buckets: VectorArray = VectorArray(self.table_capacity, object)  # this is the array object - with all the attributes and methods.
        self.current_load_factor: LoadFactor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)  # log attribute - displays current load factor
        self.max_load_factor = max_load_factor # prevents the table from exceeding this capacity
        self._hash_code = hash_code
        self._compress_func = compress_func

        # initialize each bucket as None.
        for i in range(self.table_capacity):
            self.buckets.array[i] = None

        self._hashconfig: HashFuncConfig = HashFuncConfig(self.table_capacity)

    @property
    def return_keys(self) -> VectorArray[iKey]:
        """returns key() objects for comparison (max, min, sorted etc)"""
        if self._table_keytype is None:
            found_keys = VectorArray(self.bucket_capacity, object)
        else:
            found_keys = VectorArray(self.bucket_capacity, iKey)
        table = self.buckets.array
        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size):
                    # must access the .array attribute (where the array items are stored....)
                    kv_pair = bucket.array[i]
                    k, v = kv_pair  # destructure tuple
                    found_keys.append(k)
        return found_keys

    # ----- Utility -----
    def collisions_per_bucket(self):
        """
        Returns the current number of collisions that have occured per bucket as a tuple. 
        WARNING: this resets everytime the table rehashes.
        """
        collisions = VectorArray(self.table_capacity, tuple)
        table = self.buckets.array
        # iterate over table and append index and the bucket collisions to a tuple.
        for i, bucket in enumerate(table):
            if bucket is not None and bucket.size > 0:
                bucket_collisions = max(0, bucket.size - 1)
                collisions.append((i, bucket_collisions))
        return collisions # return list of tuples

    def performance_profile_report(self):
        """Tracks the performance of the Hash table, load factor, collisions, rehashes, rehash time, capacity etc..."""
        # ANSI color codes for fun.
        BLUE = "\033[1;36m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"

        def color(input, color):
            string = f"{RESET}{color}{input}{RESET}"
            return string

        # profile stats
        # load factor - with color that changes to warn end user.
        load_factor_number = color(f"{self.current_load_factor:.2f}", GREEN) if self.current_load_factor < 0.65 else color(f"{self.current_load_factor:.2f}", RED)
        load_factor_string = f"Load Factor: {load_factor_number}"

        # total collisions - with color change
        collision_threshold: float = 0.13  # percentage boundary (13%)
        collisions_number = color(f"{self.total_collisions}", GREEN) if self.total_collisions / self.table_capacity < collision_threshold else color(f"{self.total_collisions}", RED)
        total_coll_string = f"Total Collisions: {collisions_number}"
        stats_string = f"{load_factor_string}, {total_coll_string}, Current Capacity: {self.total_elements}/{self.table_capacity}, Total Buckets Created: {self.total_buckets}"

        # rehashes
        rehashes_string = f"Total Rehashes: {self.total_rehashes}, Rehash Time (total): {self.total_rehash_time:.1f} secs, Rehash Time (latest): Completed in {self.current_rehash_time:.2f} secs"

        # per bucket stats
        collisions = self.collisions_per_bucket()
        collision_data = [col for index, col in collisions]    # collects all the collisions in the list for data analysis
        average_collisions = sum(collision_data) / len(collision_data) if collision_data else 0
        max_collisions = max(collision_data) if collision_data else 0
        min_collisions = min(collision_data) if collision_data else 0
        per_bucket_stats_string = f"Per Bucket Stats: (Reset after every rehash): Average: {average_collisions:.1f}  Max: {max_collisions} Min: {min_collisions}"

        # final profile string to print.
        profile = f"""
        {stats_string},
        {rehashes_string}, 
        {per_bucket_stats_string}
        """

        return profile

    def visualize_table(self, columns: int=16, cell_width:int = 11, row_padding: int = 3):
        """Visualizes the hash table as a console cell grid. contains the index number and number of keys in each bucket for clarity."""
        return self._utils.view_chaining_table(columns, cell_width, row_padding)

    # ----- Python Built in Overrides -----
    def __str__(self) -> str:
        return self._desc.str_chain_hashtable()

    def __repr__(self) -> str:
        return self._desc.repr_chain_hashtable()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)

    def __delitem__(self, key):
        result = self.remove(key)
        if result is None: # key not found
            raise KeyInvalidError(f"Error: Couldn't find the key to remove the element from the hash table.")

    # ----- Table Rehashing -----
    def _rehash_table(self):
        """
        dynamically resizes the hash table when it reaches a certain capacity defined by max load factor
        Step 1: Create a new array with double capacity & prime number.
        Step 2: Store the old array for copying items over
        Step 3: Recompute prime, scale, shift for the MAD compression function
        Step 4: reset the current table, the size tracker and capacity
        Step 5: Copy all the old items from the old array to the new array - via put()
        """
        start_time = time.perf_counter()

        # old array and capacity.
        old_buckets = self.buckets
        old_capacity = self.table_capacity

        # new array & capacity
        new_capacity = self._utils.find_next_prime_number(old_capacity * self.resize_factor)
        # recompute the config file (for MAD compress func prime etc)
        self._hashconfig.recompute(new_capacity)

        # create new array and capacity
        new_buckets = VectorArray(new_capacity, object)
        for i in range(new_capacity):
            new_buckets.array[i] = None

        # reset current size (will increment as we copy items over to new array)
        self.total_elements = 0
        # update to new table
        self.buckets = new_buckets
        self.table_capacity = new_capacity
        self.total_buckets = 0

        # copy items over with internal put method.
        old_table = old_buckets.array

        for old_bucket in old_table:
            if old_bucket is not None:  # check bucket not empty
                # iterate through bucket and get kv pair - add to new table via put()
                for i in range(old_bucket.size):
                    kv_pair = old_bucket.array[i]
                    key, value = kv_pair
                    self._internal_put(key, value)  # add to new table

        self.total_rehashes += 1    # update total rehashes
        end_time = time.perf_counter()

        # performance tracking
        rehash_time = end_time - start_time
        self.current_rehash_time = rehash_time  # updates current rehash time
        self.total_rehash_time += rehash_time   # updates lifetime tracker of rehash time.

    def _internal_put(self, key, value):
        """Internal put() method - does not have rehash condition"""
        # notice we dont validate the key or value again, because we dont want to wrap it again in a key object.
        # collect index via hash function
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        table = self.buckets.array
        target_bucket = table[index]
        kv_pair = (key, value)

        # if bucket doesnt exist - create a collision bucket array. add key value pair in the first slot.
        if target_bucket is None:
            new_bucket = VectorArray(self.bucket_capacity, tuple)
            new_bucket.append(kv_pair)   # add key value pair to the end if the array (which should be the beginning)
            table[index] = new_bucket  # store newly created bucket in the table array.
            self.total_elements += 1
            self.total_buckets +=1
            self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
            return # stops us from repeating recursively

        # if it already exists. - iterate through all the kv pairs in the bucket. if there is match update the kv pair.
        for i in range(target_bucket.size):
            k, v = target_bucket.array[i] # destructure the key value pair tuple inside the bucket index
            if k == key:    
                target_bucket.array[i] = kv_pair  # update
                return  # stops us from incrementing size when no new kv pair added.

        # default condition (collision) - bucket exists - but no key match found
        target_bucket.append(kv_pair)
        self.total_elements += 1
        self.total_collisions += 1
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)

    # ----- Canonical ADT Operations -----
    def put(self, key, value):
        """
        Insert a key value pair into the hash table:
        Step 1: Compute bucket index using hash function
        Step 2: Condition: bucket doesnt exist... create a BucketArray to hold key-value pairs and add key value pair to the bucket.
        Step 3: Condition: If a bucket exists... Iterate the bucket to see if the key already exists → if so, Overwrite value, Otherwise, append the new key-value pair.
        Step 4: Condition: Bucket exists, but key doesnt exist. append the key value pair to the bucket.
        Step 4: Increment size tracker
        Step 5: Check load factor → call _rehash_table if exceeded.
        """
        # rehash table if exceed max load factor (+1 for future insertion) -- needs to be first.
        if (self.total_elements + 1) / self.table_capacity > self.max_load_factor:
            self._rehash_table()

        key = Key(key)
        self._utils.check_key_type(key)  # ensures the input matches table key type.
        value = TypeSafeElement(value, self.datatype)

        # collect index via hash function
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        table = self.buckets.array
        target_bucket = table[index]
        kv_pair = (key, value)

        # if bucket doesnt exist - create a new bucket array. add key value pair in the first slot.
        if target_bucket is None:
            new_bucket = VectorArray(self.bucket_capacity, tuple)
            new_bucket.append(kv_pair)   # add key value pair to the end if the array (which should be the beginning)
            table[index] = new_bucket  # store newly created bucket in the table array.
            self.total_elements += 1
            self.total_buckets += 1
            self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
            return # stops us from repeating recursively

        # if it already exists. - iterate through all the kv pairs in the bucket. if there is match update the kv pair.
        for i in range(target_bucket.size):
            k, v = target_bucket.array[i] # destructure the key value pair tuple inside the bucket index
            if k == key:    
                target_bucket.array[i] = kv_pair  # update
                return  # stops us from incrementing size when no new kv pair added.

        # default condition - bucket exists (colllision) - but no key match found
        target_bucket.append(kv_pair)
        self.total_elements += 1
        self.total_collisions += 1
        self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)

    def get(self, key, default: Optional[T] = None):
        """
        Searches for a key & returns the value. Optionally returns a default value if desired.
        Compute the bucket index using hash function
        Check the bucket array for the key.
        Return the value if found. If not found, optionally return a default value.
        """
        key = Key(key)
        self._utils.check_key_type(key)  # ensures the input matches table key type.

        if default is not None:
            default = TypeSafeElement(default, self.datatype)

        # compute index via hash function
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        table = self.buckets.array
        target_bucket = table[index]  # locate bucket.

        # if target bucket exists - iterate through and search for key
        if target_bucket is not None:
            for i in range(target_bucket.size):
                k, v = target_bucket.array[i]
                if k == key:
                    return v

        return default if default is not None else None

    def remove(self, key):
        """
        Compute the bucket index using hash function
        Check the bucket array for the key.
        Remove the key-value pair if found & return the value
        """
        key = Key(key)
        self._utils.check_key_type(key)  # ensures the input matches table key type.

        # compute index via hash function
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        table = self.buckets.array
        target_bucket = table[index]

        # if there is no target bucket, do nothing
        if target_bucket is None:
            return None

        # otherwise - iterate over the target bucket and search for key. If found, delete the key. and return it
        for i in range(target_bucket.size):
            k, v = target_bucket.array[i]
            if k == key:
                deleted_kv_pair = target_bucket.delete(i)
                del_key, del_value = deleted_kv_pair
                self.total_elements -= 1
                self.current_load_factor = self._utils.calculate_load_factor(self.total_elements, self.table_capacity)
                return del_value

        return None # nothing found

    def keys(self):
        """Return a array of all the keys in the hash table"""
        if self._table_keytype is None:
            found_keys = VectorArray(self.bucket_capacity, object)
        else:
            found_keys = VectorArray(self.bucket_capacity, self._table_keytype)

        table = self.buckets.array
        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size): 
                    # must access the .array attribute (where the array items are stored....)
                    kv_pair = bucket.array[i]
                    k,v = kv_pair   # destructure tuple
                    k = k.value
                    found_keys.append(k)
        return found_keys

    def values(self):
        """Return a array of all the values of the hash table"""
        found_values = VectorArray(self.total_elements, self.datatype)
        table = self.buckets.array
        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size):
                    # must access the .array attribute (where the array items are stored....)
                    kv_pair = bucket.array[i]
                    k, v = kv_pair  # destructure tuple
                    found_values.append(v)
        return found_values

    def items(self):
        """Returns a array of tuples of all the key value pairs in the hash table"""
        found_items = VectorArray(self.total_elements, tuple)
        table = self.buckets.array
        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size):
                    # must access the .array attribute (where the array items are stored....)
                    k,v = slot = bucket.array[i]
                    k = k.value
                    kv_pair = (k, v)
                    found_items.append(kv_pair)
        return found_items

    # ----- Meta Collection ADT Operations -----
    def __contains__(self, key):
        """Does the Hash table contain an item with the specified key?"""
        key = Key(key)
        self._utils.check_key_type(key)  # ensures the input matches table key type.
        # compute index via hash function
        hashgen = HashFuncGen(key, self._hashconfig, self._hash_code, self._compress_func)
        index = hashgen.hash_function()

        target_bucket = self.buckets.array[index]  # traverse bucket

        # bucket doesnt exist
        if target_bucket is None:
            return False

        # traverse bucket and find key.
        for index in range (target_bucket.size):
            k, v = target_bucket.array[index]
            if k == key:
                return True
        return False

    def is_empty(self):
        return self.total_elements == 0

    def __len__(self):
        """Returns the total number of key value pairs currently stored in the table"""
        return self.total_elements

    def clear(self):
        """Remove all key value pairs from the hash table."""

        self.total_elements = 0
        self.total_buckets = 0
        self.total_collisions = 0
        self.current_load_factor = 0
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_rehash_time = 0.0
        self.buckets = VectorArray(self.table_capacity, object)

        for i in range(self.table_capacity):
            self.buckets.array[i] = None

        self._hashconfig.recompute(self.table_capacity)

    def __iter__(self):
        """iterates over the hash table via generator - useful for looping and ranges..."""
        table = self.buckets.array
        for bucket in table:
            if bucket is not None:
                for i in range(bucket.size):
                    kv_pair = bucket.array[i]
                    k, v = kv_pair
                    k = k.value # unpack key
                    yield k

    # Main ---- Client Facing Code -----

    # todo add batch insert, batch update, batch delete
    # todo merge hash tables together.
    # todo clean up test harness. (utilize random sample etc.)
    # todo clean up visualization and reporting
    # todo check __contains__ logic.


class StressTestHashTable():
    """A suite of tests to randomly test the hash table data structure..."""

    AI = type("ArtificialPerson",(),
        {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__str__": lambda self: f"NotAPerson({self.name})",
            "__repr__": lambda self: f"NotAPerson({self.name})",
        },
    )

    TEST_CLASS = AI("waragoobafffffucccccdddddd")

    def __init__(self, input_list, number_of_items, datatype, table_size) -> None:
        # input data
        self.input_list = input_list
        self.datatype = datatype
        self.number_of_items = number_of_items
        # generate keys, values & items
        self.test_values = self._generate_stress_test_list(self.input_list, self.number_of_items)
        self.test_keys = [f"Key_{i}" for i in range(self.number_of_items)]
        self.test_items = list(zip(self.test_keys, self.test_values))

        # initialize table.
        print(f"=== ChainHashTable Test: using type: {datatype.__name__} ===")
        self.initial_table_size = max(3, table_size)
        self.hashtable = ChainHashTable(datatype, self.initial_table_size)
        print(f"\nInitialized Table:\n{self.hashtable}")

    # ---------------- Utility ----------------
    def _generate_stress_test_list(self, preset_list, max_number_of_items):
        """takes a preset list - and generates a new list, wrapping the items in the list around like a clock until the max number of items is reached."""
        return [preset_list[i % len(preset_list)] for i in range(max_number_of_items)]

    def _infostring(self):
        print(f"{self.hashtable}")

    # ---------------- Operations ----------------
    def test_is_empty(self):
        print(f"\n=== Testing Is Empty() ===")
        print(f"Is the table empty? Result: {self.hashtable.is_empty()}")
        self._infostring()

    def test_insertion(self, subdivide=None):
        # testing put() - insertion.
        subdivide = subdivide if subdivide is not None else 5
        print(f"\n=== Testing Insertion Put() ===")
        items = self.test_items[:len(self.test_items)//subdivide]
        print(f"We will add: {len(items)} elements to the hash table.")
        info_step = max(1, len(items)//5)

        # loop through subsection of data.
        for i, item in enumerate(items):
            # add kv pair to table
            key, value = item
            self.hashtable.put(key, value)

            # prints after a specific fraction of inserts have occured, like 1/5 etc
            if (i+1) % info_step == 0:
                print(f"Adding Elements....{i+1}/{len(items)}")
                print(self.hashtable.performance_profile_report())

    def test_type_safety(self):
        print(f"\n=== Testing Type Enforcement ===")
        wrong_type = StressTestHashTable.TEST_CLASS

        print(f"Testing Valid Type...")
        key = "valid_type"
        value = self.input_list[0]
        try:
            self.hashtable.put(key, value)
            print(f"Hashtable contains key?: {key in self.hashtable}")
        except Exception as error:
            print(f"Valid Type rejected!: {error}")

        print(f"Testing Invalid Type...")
        try:
            self.hashtable.put("wrong_type", wrong_type)
        except Exception as error:
            print(f"Invalid Type rejected: {error}")
            print(f"Hashtable contains key?: {'wrong type' in self.hashtable}")

    def test_get_and_set(self, subdivide=None):
        print(f"\n=== Testing __getitem__ & __setitem__ ===")
        subdivide = subdivide if subdivide is not None else 2

        items = list(self.hashtable.items()) 
        length = len(items)
        changes = length // subdivide
        group_a = items[:changes]   # grabs a subset from the start of items.
        group_b = items[-changes:]  # grabs a subset from the end of items
        alterations = 0
        info_step = max(1, length // 10)
        print(f"Logging a small subsection of the alterations for console...")

        for i, ((key_a, value_a), (key_b, value_b)) in enumerate(zip(group_a, group_b)):
            # __getitem__
            getitem = self.hashtable[key_a]
            assert getitem == value_a, f"Error: Expected: {value_a} Got: {getitem}"
            # __setitem__
            self.hashtable[key_a] = value_b
            assert self.hashtable[key_a] == value_b, f"Error: Expected: {value_b} Got: {self.hashtable[key_a]}"
            alterations += 1
            if i % info_step == 0:
                print(f"Got: {getitem} & Set Value to: {self.hashtable[key_a]}")
        print(f"\nAlterations: {alterations} made via set()")
        print(self.hashtable.performance_profile_report())

    def test_remove(self, subdivide=None):
        print(f"\n=== Testing remove() ===")
        subdivide = subdivide if subdivide is not None else 3
        items = list(self.hashtable.items())
        remove_subset = items[: len(items) // subdivide]
        print(f"Removing: {len(remove_subset)} elements from our hash table")

        print(self.hashtable.performance_profile_report())
        for key, value in remove_subset:
            self.hashtable.remove(key)
            # assert key not in [k for k, v in self.hashtable.items()], f"Error: Assertion for remove({key} failed. investigate further."
        print(self.hashtable.performance_profile_report())

    def test_contains(self, key=None):
        print(f"\n=== Testing contains() ===")
        keys = self.hashtable.keys()
        test_key = key if key is not None else random.choice(keys)
        print(f"Testing if Hash table contains {test_key} == {test_key in self.hashtable}")
        try:
            assert test_key in self.hashtable, f"Error: Assertion for Contains({test_key}) failed!"
        except AssertionError as error:
            print(f"{error}")

    def test_iteration_keys_values_items(self):
        print(f"\n=== Testing Keys(), values() & items() and general iteration===")
        keys = self.hashtable.keys()
        values = self.hashtable.values()
        items = self.hashtable.items()
        assert len(keys) == len(values) == len(items), f"Error: mismatch between the length of {keys}, {values} & {items}"
        print(f"Checking that the elements in items() are found in keys() & values()")

    def test_clear(self):
        print(f"\n=== Testing Clear() ===")
        print(self.hashtable.performance_profile_report())
        self.hashtable.clear()
        # verify empty
        assert self.hashtable.is_empty(), "Error: Table should be empty after clear()"
        assert len(self.hashtable) == 0, f"Error: Expected size 0, got {len(self.hashtable)}"
        assert list(self.hashtable.keys()) == [], "Error: Keys() not empty after clear()"
        assert list(self.hashtable.values()) == [], "Error: Values() not empty after clear()"
        assert list(self.hashtable.items()) == [], "Error: Items() not empty after clear()"
        assert self.hashtable.total_collisions == 0, f"Error: Total Collisions should be 0 Got: {self.hashtable.total_collisions}"
        print(self.hashtable.performance_profile_report())
        self._infostring()
        print(f"Clear() successful: table empty, size reset, no keys/values/items")

    # ---------------- Presets ----------------
    def normal_test(self):
        """Low key normal testing - just making sure everything works."""
        start_time = time.perf_counter()
        self.test_is_empty()
        self.test_insertion()
        self.test_get_and_set()
        self.test_remove()
        self.test_type_safety()
        self.test_iteration_keys_values_items()
        self.test_clear()
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        print(f"Normal low key Test Completed in {minutes} min & {seconds:.2f} secs")

    def stress_test(self):
        print(f"\n=== STRESS TEST MAX CAPACITY...===")
        start_time = time.perf_counter()
        self.test_is_empty()
        self.test_insertion(subdivide=1)
        self.test_type_safety()
        self.test_get_and_set(subdivide=2)
        self.test_remove(subdivide=10)
        self.test_contains()
        self.test_contains("gfdhugfdg")
        self.test_iteration_keys_values_items()
        self.hashtable.visualize_table()
        self.test_clear()
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        print(f"Stress Test Completed in {minutes} min & {seconds:.2f} secs")


def main():
    # region inputdata
    # ------------- Input Data -----------------
    # # Strings
    preset_strings = [
    "apple", "banana", "cherry", "date", "elderberry",
    "fig", "grape", "honeydew", "kiwi", "lemon",
    "mango", "nectarine", "orange", "papaya", "quince"
    ]

    # Integers
    preset_ints = [0, 1, -1, 42, -42, 100, 999, -999, 123456, -123456]

    # Floats
    preset_floats = [0.0, 1.5, -1.5, 3.14159, -3.14159, 2.71828, -2.71828, 1234.5678, -9876.54321]

    # Booleans
    preset_bools = [True, False]

    # Dictionaries
    preset_dicts = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
    {"color": "red", "value": "#FF0000"},
    {"color": "green", "value": "#00FF00"}
    ]

    # Lists
    preset_lists = [[1, 2, 3],["a", "b", "c"],[True, False, True],[42, 43, 44],[]]

    # Tuples
    preset_tuples = [(1, 2, 3),("x", "y", "z"),(True, False, True),(42, 43, 44),()]
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
    person_names = ["Alice", "Bob", "Charlie", "Diana", "Eve","Frank", "Grace", "Hank", "Ivy", "Jack"]
    preset_dynamic_objects = [Person(name) for name in person_names]
    # endregion

    # ------------- Utilize Test Suite -----------------
    # actual stress test list.
    normal_num_of_items = 100
    stress_number_of_items = 100
    test = StressTestHashTable(preset_dynamic_objects, stress_number_of_items, Person, table_size=stress_number_of_items // 6)
    test.stress_test()

    # items_list = [preset_list[i % len(preset_list)] for i in range(max_number_of_items)]


if __name__ == "__main__":
    main()
