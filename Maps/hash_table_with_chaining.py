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
    Tuple
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
    def get(self, key: str, default: Optional[T]) -> Optional[T]:
        """retrieves a key value pair from the hash table, with an optional default if the key is not found."""
        pass

    @abstractmethod
    def remove(self, key: str) -> Optional[T]:
        """removes a key value pair from the hash table."""
        pass

    @abstractmethod
    def keys(self) -> Optional['BucketArray']:
        """Return a set of all the keys in the hash table"""
        pass

    @abstractmethod
    def values(self) -> Optional["BucketArray"]:
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
    """Dynamic Array — automatically resizes as elements are added."""

    def __init__(self, capacity: int, datatype: type, datatype_map: dict = CTYPES_DATATYPES) -> None:

        # datatype
        self.datatype = datatype
        self.datatype_map = datatype_map
        self.type = None

        # Core Array Properties
        self.min_capacity = max(4, capacity)  # sets a minimum size for the array
        self.capacity = capacity  # sets the total amount of spaces for the array
        self.size = 0  # tracks the number of elements in the array
        # creates a new ctypes/numpy array with a specified capacity
        self.array = self._initialize_new_array(self.capacity)  

    # ----- Utility -----
    def _initialize_new_array(self, capacity):
        """chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)"""
        if self.datatype_map == CTYPES_DATATYPES:
            new_array = self._init_ctypes_array(capacity)
            return new_array
        elif self.datatype_map == NUMPY_DATATYPES:
            new_array = self._init_numpy_array(capacity)
            return new_array
        else:
            raise ValueError(f"Datatype Map Unknown... Map: {self.datatype_map}")

    def _init_ctypes_array(self, capacity):
        """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
        # setting ctypes datatype -- needed for the array. (object is a general all purpose datatype)
        if self.datatype not in self.datatype_map:
            self.type = ctypes.py_object  # general all purpose python object
        else:
            self.type = CTYPES_DATATYPES[self.datatype]  # maps type of array to ctype
        # creates a class object - an array of specified number of a specified type
        dynamic_array_cls = self.type * capacity
        # initializes array with preallocated memory block
        new_ctypes_array = dynamic_array_cls()
        return new_ctypes_array

    def _init_numpy_array(self, capacity):
        """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
        if self.datatype not in NUMPY_DATATYPES:
            self.type = object  # general all purpose python object.
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            new_numpy_array = numpy.empty(capacity, dtype=self.type)
        return new_numpy_array

    def _enforce_type(self, value):
        """type enforcement - checks that the value matches the prescribed datatype."""
        if not isinstance(value, self.datatype):
            raise TypeError(
                f"Error: Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}"
            )

    def _index_boundary_check(self, index, is_insert: bool = False):
        """
        Checks that the index is a valid number for the array.
        index needs to be greater than 0 and smaller than the number of elements (size)
        """
        if is_insert:
            if index < 0 or index > self.capacity:
                raise IndexError("Error: Index is out of bounds.")
        else:
            if index < 0 or index >= self.capacity:
                raise IndexError("Error: Index is out of bounds.")

    def _grow_array(self):
        """
        Step 1: Store existing array data and capacity.
        Step 2: Initialize new array with * 2 capacity
        Step 3: Copy old items to new array
        Step 4: Update the capacity to reflect the new extended capacity.
        Step 5: return the array for use in the program.
        """
        old_array = self.array
        old_capacity = self.capacity
        new_capacity = self.capacity * 2
        new_array = self._initialize_new_array(new_capacity)
        for i in range(self.size):
            new_array[i] = old_array[i]
        self.capacity = new_capacity
        return new_array

    def _shrink_array(self):
        """Shrink array in half"""
        old_array = self.array
        old_capacity = self.capacity
        new_capacity = max(self.min_capacity, self.capacity // 2)
        new_array = self._initialize_new_array(new_capacity)
        for i in range(self.size):
            new_array[i] = old_array[i]
        self.capacity = new_capacity
        return new_array

    def __str__(self) -> str:
        """a list of strings representing all the elements in the array"""
        items = ", ".join(str(self.array[i]) for i in range(self.size))
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        # f"[{items}], Capacity: {self.size}/{self.capacity}, Type: {string_datatype}"
        return f"[{string_datatype}][{items}]"

    def __getitem__(self, index):
        """Built in overrid - adds indexing"""
        return self.get(index)

    def __setitem__(self, index, value):
        """Built in override - adds indexing."""
        self.set(index, value)

    # ----- Canonical ADT Operations -----
    def get(self, index):
        """Return element at index i"""
        self._index_boundary_check(index)
        result = self.array[index]
        return cast(T, result)

    def set(self, index, value):
        """Replace element at index i with x"""
        self._enforce_type(value)
        self._index_boundary_check(index)
        self.array[index] = value

    def insert(self, index, value):
        """
        Insert x at index i, shift elements right:
        Step 1: Loop through elements: Start at the end & go backwards. Stop at the index element (where we want to insert.)
        Step 2: copy element from the previous index (the left) - this shifts every element to the right.
        Step 3: Now the target index will contain a duplicate value - which we will overwrite with the new value
        Step 4: Increment Array Size Tracker
        """

        self._enforce_type(value)
        self._index_boundary_check(index, is_insert=True)
        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array()

        # if index value is the end of the array - utilize O(1) append
        if index == self.size:
            self.append(value)
            return

        # move all array elements right.
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]  # (e.g. elem_4 = elem_3)
        self.array[index] = value
        self.size += 1  # update size tracker

    def delete(self, index):
        """
        Remove element at index i, shift elements left:
        Step 1: store index to return later (the deleted item)
        Step 2: Loop through elements from the index to the end of the array.
        Step 3: copy element from the future index (the right). This shifts each element left (from the target index point onwards.)
        Step 4: For the last element in the array, change value to None
        Step 5: decrement the size tracker.
        Step 6: return deleted value
        """

        if self.is_empty():
            raise ValueError("Error: Array is Empty.")

        self._index_boundary_check(index)

        # dynamically shrink array if capacity at 25% and greater than min capacity
        if self.size == self.capacity // 4 and self.capacity > self.min_capacity:
            self.array = self._shrink_array()

        deleted_value = self.array[index]  # store index for return
        # shift elements left -- Starts from the deleted index (Goes Backwards)
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]  # (elem4 = elem5)
        if self.type is object or self.type is ctypes.py_object:
            self.array[self.size - 1] = (
                None  # removes item from the end of the stored items
            )
        self.size -= 1  # decrement size tracker
        return deleted_value

    def append(self, value):
        """Add x at end -- O(1)"""
        self._enforce_type(value)

        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array()

        self.array[self.size] = value
        self.size += 1

    def prepend(self, value):
        """Insert x at index 0 -- O(N) - Same logic as insert, shift elems right"""
        self._enforce_type(value)
        # dynamically resize the array if capacity full.
        if self.size == self.capacity:
            self.array = self._grow_array()

        for i in range(self.size, 0, -1):
            self.array[i] = self.array[i - 1]

        self.array[0] = value
        self.size += 1

    def index_of(self, value):
        """Return index of first x (if exists)"""
        for i in range(self.size):
            if self.array[i] == value:
                return i
        return None

    # ----- Meta Collection ADT Operations -----
    def __len__(self):
        """Return number of elements"""
        return self.size

    def is_empty(self):
        """returns true if sequence is empty"""
        return self.size == 0

    def clear(self):
        """removes all items and reinitializes a new array with the original capacity, resets the size tracker also"""
        self.array = self._initialize_new_array(self.min_capacity)
        self.capacity = self.min_capacity
        self.size = 0

    def __contains__(self, value):
        """True if x exists in sequence"""
        for i in range(self.size):
            if self.array[i] == value:
                return True
        return False

    def __iter__(self):
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        for i in range(self.size):
            result = self.array[i]
            yield cast(T, result)


class ChainHashTable(MapADT[T]):
    """
    Hash Table implementation with collision chaining via array bucket.
    Essentially we build a Multi Dimensional Array. (MD Array) - the first array stores bucket arrays. the bucket arrays store kv pairs
    The keys must be unique strings, the values can be enforced to be a specific type.
    Uses Experimental ** exponential resizing for the table rehashing -
    has controls for max load factor and resize factor.
    """
    def __init__(self, datatype: type, table_capacity: int = 10, max_load_factor: float = 0.8, resize_factor: int = 10) -> None:
        # values datatype enforcement.
        self.datatype = datatype
        # trackers
        self.total_elements = 0   # number of key-value pairs
        self.total_buckets = 0  # number of created buckets.
        self.total_collisions = 0 # tracks the number of collisions that have occured
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_rehash_time = 0.0

        # core attributes
        self.resize_factor = resize_factor
        self.table_capacity = self._find_next_prime_number(table_capacity)    # number of slots in hash table
        self.bucket_capacity = self._find_next_prime_number(10) # initializes each bucket with this number of slots.
        self.buckets = BucketArray(self.table_capacity, object)  # this is the array object - with all the attributes and methods.
        self.current_load_factor = self._calculate_load_factor()  # log attribute - displays current load factor
        self.max_load_factor = max_load_factor # prevents the table from exceeding this capacity
        # initialize each bucket as None.
        for i in range(self.table_capacity):
            self.buckets.array[i] = None

        # MAD attributes - fixed after initialization (until table rehashing)
        self.prime = self._find_next_prime_number(self.table_capacity)  # just slightly above table size.
        # must be smaller than prime attribute. (and cannot be a cofactor so cannot be 1)
        self.scale = random.randint(2, self.prime-1)    
        self.shift = random.randint(2, self.prime-1)
        # add to a packed tuple for convenience
        self.mad_modifiers = (self.prime, self.scale, self.shift)

    # ----- Utility -----
    def currently_used_indexes(self):
        indexes = BucketArray(self.bucket_capacity, int)
        table = self.buckets.array
        for i, bucket in enumerate(table):  # access the underlying array
            if bucket is not None and bucket.size > 0:
                indexes.append(i)
        return indexes

    def collisions_per_bucket(self):
        """Returns the current number of collisions that have occured per bucket as a tuple. WARNING: this resets everytime the table rehashes."""
        collisions = BucketArray(self.table_capacity, tuple)
        table = self.buckets.array
        # iterate over table and append index and the bucket collisions to a tuple.
        for i, bucket in enumerate(table):
            if bucket is not None and bucket.size > 0:
                bucket_collisions = max(0, bucket.size - 1)
                collisions.append((i, bucket_collisions))

        return collisions # return list of tuples

    def per_bucket_collisions_report(self, collisions: BucketArray):
        string_collisions = ", ".join(f'Bucket: {index}, Coll: {collides}' for index, collides in collisions)
        infostring = f"\nCurrent Collisions per Bucket (resets after rehash): {string_collisions}.\nAggregated (doesn't reset) Total: {self.total_collisions} Collisions\n"
        return infostring

    def convert_to_minutes(self, time):
        minutes = int(time // 60)
        seconds = time % 60
        return minutes, seconds

    def performance_profile_report(self):
        """Tracks the performance of the Hash table, load factor, collisions, rehashes, rehash time, capacity etc..."""
        total_minutes, total_seconds = self.convert_to_minutes(self.total_rehash_time)
        current_minutes, current_seconds = self.convert_to_minutes(self.current_rehash_time)
        collisions = self.collisions_per_bucket()
        collision_data = [col for index, col in collisions]    # collects all the collisions in the list for data analysis
        # mean
        average_collisions = sum(collision_data) / len(collision_data) if collision_data else 0
        max_collisions = max(collision_data) if collision_data else 0
        min_collisions = min(collision_data) if collision_data else 0

        profile = f"""
        Load Factor: {self.current_load_factor:.2f}, Total Collisions: {self.total_collisions}, Current Capacity: {self.total_elements}/{self.table_capacity}, Total Buckets Created: {self.total_buckets},
        Total Rehashes: {self.total_rehashes}, Rehash Time (total): {self.total_rehash_time:.1f} secs, Rehash Time (latest): Completed in {self.current_rehash_time:.2f} secs, 
        Per Bucket Stats: (Reset after every rehash): Average: {average_collisions:.1f}  Max: {max_collisions} Min: {min_collisions}
        """
        return profile

    def visualize_table(self, columns: int=8, cell_width:int = 20, row_padding: int = 3):
        """Visualizes the hash table as a console cell grid. contains the index number and number of keys in each bucket for clarity."""
        table = self.buckets.array
        table_container = []

        # table creation.
        columns = columns
        cell_width = cell_width
        row_seperator = "-" * (columns * (cell_width + row_padding))

        # loops through every bucket in the table. appends the index number and count of keys for each bucket with items.
        # otherwise appends an empty list. (we will fill this in later with placeholder text.)
        for idx, bucket in enumerate(table):
            bucket_container = []
            if bucket is None:
                table_container.append([])
            if bucket is not None:
                count = len(bucket) if bucket else 0 # type: ignore
                stats = f"Idx: {idx}: keys: {count}"
                bucket_container.append(stats)  # append found items to the bucket container
                table_container.append(bucket_container)    # append buckets to the table container.

        # rows logic --
        table_size = len(table_container)
        for i in range(0, table_size, columns):
            row = table_container[i:i+columns]  # slices table container to create a sublist for each row of size columns.
            row_display = []
            # for every bucket in the sliced part of the table - if its empty append a placeholder, otherwise append the stats text
            for bucket in row:
                if not bucket:  # if the bucket is empty (the list representation of a bucket)
                    row_display.append("[]".center(cell_width))
                else:
                    row_display.append(", ".join(str(stats) for stats in bucket).center(cell_width))
            print(row_seperator)
            print(f"{' | '.join(row_display)}")
            print(row_seperator)

    # ----- Python Built in Overrides -----
    def __str__(self) -> str:
        items = self.items()
        infostring = f"[{self.datatype.__name__}]{{{{{str(', '.join(f'{k}: {v}' for k, v in items))}}}}}"
        return infostring

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)

    # ----- Hash Function -----
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
        # M-A-D Method core logic
        multiply = self.scale * hash_code
        add = multiply + self.shift
        divide = add % self.prime
        index = divide % self.table_capacity  # finally mod by table capacity
        return index

    def _k_mod_compression_function(self, hash_code, table_capacity):
        """Takes a hash code and conforms it to the hash table size, and returns the index number"""
        # the division method: aka k-mod
        k_mod = hash_code % table_capacity
        return k_mod

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
        bit_mask = 2**64-1  # This creates a 64-bit mask
        hash_code = 0
        # horner's method = hash * prime + char(ascii number)
        for character in key:
            hash_code = hash_code * prime_weighting + ord(character) & bit_mask
        hash_code ^= (hash_code << shift) & bit_mask
        hash_code ^= (hash_code >> shift)
        hash_code ^= hash_code << (shift // 2) & bit_mask

        return hash_code & bit_mask

    def _hash_function(self, key: str):
        """Combines the hash code and compression function and returns an index value for a key."""
        poylnomial_hashcode = self._polynomial_hash_code(key)   # better for smaller tables like up to 1000 (0 collisions)
        cyclic_shift_hashcode = self._cyclic_shift_hash_code(key)   # better for huge tables like 10,000+ (1000 collisions)
        combined_hashcode = self._cyclic_polynomial_combo_hash_code(key)
        index = self._mad_compression_function(cyclic_shift_hashcode)
        return index

    # ----- Table Rehashing -----
    def _calculate_load_factor(self) -> float:
        """calculates the load factor of the current hashtable"""
        return self.total_elements / self.table_capacity

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
        new_capacity = self._find_next_prime_number(old_capacity * self.resize_factor)
        # new MAD modifiers
        new_prime = self._find_next_prime_number(new_capacity) # self.table_capacity * 300000000
        new_scale = random.randint(2, new_prime - 1)
        new_shift = random.randint(2, new_prime - 1)

        # create new array and capacity
        new_buckets = BucketArray(new_capacity, object)
        for i in range(new_capacity):
            new_buckets.array[i] = None

        # reset current size (will increment as we copy items over to new array)
        self.total_elements = 0
        # update to new table
        self.buckets = new_buckets
        self.prime = new_prime
        self.scale = new_scale
        self.shift = new_shift
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
        rehash_time = end_time - start_time
        self.current_rehash_time = rehash_time  # updates current rehash time
        self.total_rehash_time += rehash_time   # updates lifetime tracker of rehash time.

    def _internal_put(self, key, value):
        """Internal put() method - does not have rehash condition"""
        self._enforce_type(value)
        index = self._hash_function(key)

        table = self.buckets.array
        target_bucket = table[index]
        kv_pair = (key, value)

        # if bucket doesnt exist - create a collision bucket array. add key value pair in the first slot.
        if target_bucket is None:
            new_bucket = BucketArray(self.bucket_capacity, tuple)
            new_bucket.append(kv_pair)   # add key value pair to the end if the array (which should be the beginning)
            table[index] = new_bucket  # store newly created bucket in the table array.
            self.total_elements += 1
            self.total_buckets +=1
            self.current_load_factor = self._calculate_load_factor()
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
        self.current_load_factor = self._calculate_load_factor()

    # ----- Validation Checks & Errors -----
    def _enforce_type(self, value):
        """type enforcement - checks that the value matches the prescribed datatype."""
        if not isinstance(value, self.datatype):
            raise TypeError(f"Error: Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}")

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

        self._enforce_type(value)

        # rehash table if exceed max load factor (+1 for future insertion)
        if (self.total_elements + 1) / self.table_capacity > self.max_load_factor:
            self._rehash_table()

        # collect index via hash function
        index = self._hash_function(key)

        table = self.buckets.array
        target_bucket = table[index]
        kv_pair = (key, value)

        # if bucket doesnt exist - create a new bucket array. add key value pair in the first slot.
        if target_bucket is None:
            new_bucket = BucketArray(self.bucket_capacity, tuple)
            new_bucket.append(kv_pair)   # add key value pair to the end if the array (which should be the beginning)
            table[index] = new_bucket  # store newly created bucket in the table array.
            self.total_elements += 1
            self.total_buckets += 1
            self.current_load_factor = self._calculate_load_factor()
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
        self.current_load_factor = self._calculate_load_factor()

    def get(self, key, default=None):
        """
        Searches for a key & returns the value. Optionally returns a default value if desired.
        Compute the bucket index using hash function
        Check the bucket array for the key.
        Return the value if found. If not found, optionally return a default value.
        """
        # compute index via hash function
        index = self._hash_function(key)
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
        index = self._hash_function(key)
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
                self.current_load_factor = self._calculate_load_factor()
                return del_value

        return None # nothing found

    def keys(self):
        """Return a array of all the keys in the hash table"""
        found_keys = BucketArray(self.bucket_capacity, str)
        table = self.buckets.array

        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size): 
                    # must access the .array attribute (where the array items are stored....)
                    kv_pair = bucket.array[i]
                    k,v = kv_pair   # destructure tuple
                    found_keys.append(k)

        return found_keys

    def values(self):
        """Return a array of all the values of the hash table"""
        found_values = BucketArray(self.total_elements, object)
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
        found_items = BucketArray(self.total_elements, tuple)
        table = self.buckets.array

        # iterate through table O(N*K)
        for bucket in table:
            if bucket is not None:
                # only iterate over the populated portion of the bucket (bucket.size)
                for i in range(bucket.size):
                    # must access the .array attribute (where the array items are stored....)
                    kv_pair = bucket.array[i]
                    k, v = kv_pair  # destructure tuple
                    found_items.append(kv_pair)

        return found_items

    # ----- Meta Collection ADT Operations -----
    def contains(self, key):
        """Does the Hash table contain an item with the specified key?"""

        index = self._hash_function(key)
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
        self.buckets = BucketArray(self.table_capacity, object)
        for i in range(self.table_capacity):
            self.buckets.array[i] = None

        # MAD attributes
        self.prime = self._find_next_prime_number(self.table_capacity) # self.table_capacity * 300000000
        self.scale = random.randint(2, self.prime - 1)
        self.shift = random.randint(1, self.prime - 1)

    def __iter__(self):
        """iterates over the hash table via generator - useful for looping and ranges..."""
        table = self.buckets.array
        for bucket in table:
            if bucket is not None:
                for i in range(bucket.size):
                    kv_pair = bucket.array[i]
                    k, v = kv_pair
                    yield k

    # Main ---- Client Facing Code -----

    # todo custom dependency injected hash functions per instance.
    # todo add batch insert, batch update, batch delete
    # todo merge hash tables together.

# Dynamic classes


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
        self.hashtable = ChainHashTable[datatype](datatype, self.initial_table_size, resize_factor=10)
        print(f"\nInitialized Table:\n{self.hashtable}")

    # ---------------- Utility ----------------
    def _generate_stress_test_list(self, preset_list, max_number_of_items):
        """takes a preset list - and generates a new list, wrapping the items in the list around like a clock until the max number of items is reached."""
        return [preset_list[i % len(preset_list)] for i in range(max_number_of_items)]

    def _infostring(self):
        print(f"{self.hashtable}, Indices --> {self.hashtable.currently_used_indexes()}")

    def test_prime_number_tech(self):
        # prime number test.
        print(f"\n=== Testing Prime Number Functionality ===")
        print(f"\nChecking if Initial Table Size: {self.initial_table_size} is a prime number: {self.hashtable._is_prime_number(self.initial_table_size)}")
        print(f"testing find next largest prime: Expected: actual table size") 
        print(f"Result: {self.hashtable._find_next_prime_number(self.initial_table_size)} Vs Actual Table Size {self.hashtable.table_capacity}")

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
            print(f"Hashtable contains key?: {self.hashtable.contains(key)}")
        except Exception as error:
            print(f"Valid Type rejected!: {error}")

        print(f"Testing Invalid Type...")
        try:
            self.hashtable.put("wrong_type", wrong_type)
        except Exception as error:
            print(f"Invalid Type rejected: {error}")
            print(f"Hashtable contains key?: {self.hashtable.contains('wrong_type')}")

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
        print(f"Testing if Hash table contains {test_key} == {self.hashtable.contains(test_key)}")
        try:
            assert self.hashtable.contains(test_key) == True, f"Error: Assertion for Contains({test_key}) failed!"
        except AssertionError as error:
            print(f"{error}")

    def test_hash_function(self, test_key=None):
        print(f"\n=== Testing Hash Code & Compression Function: Hash Function Independent Test...===")
        # get the hash code and compression function index for a specific item. -- then check to ensure the item is in that index.
        items = list(self.hashtable.items())

        test_item = random.choice(items)
        test_key, test_value = test_item
        # hash function
        index = self.hashtable._hash_function(test_key)
        table = self.hashtable.buckets.array
        bucket = table[index]

        print(f"Computing hash function for {test_key}: index value: {index}...")
        print(f"Now lookup item via index number: {index} Expected: {test_item}: Got: {bucket}")

    def test_iteration_keys_values_items(self):
        print(f"\n=== Testing Keys(), values() & items() and general iteration===")

        keys = self.hashtable.keys()
        values = self.hashtable.values()
        items = self.hashtable.items()

        assert len(keys) == len(values) == len(items), f"Error: mismatch between the length of {keys}, {values} & {items}"
        print(f"Checking that the elements in items() are found in keys() & values()")
        print(self.hashtable.performance_profile_report())

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
        self.test_prime_number_tech()
        self.test_is_empty()
        self.test_insertion()
        self.test_get_and_set()
        self.test_remove()
        self.test_type_safety()
        self.test_hash_function()
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
        self.test_hash_function()
        self.test_iteration_keys_values_items()
        self.hashtable.visualize_table(columns=10)
        keys = self.hashtable.keys()
        test_key = random.choice(keys)
        index = self.hashtable._hash_function(str(test_key))
        print(f"Testing table visualization:\nExpected: {test_key} at index: {index}\nGot: {self.hashtable.buckets.array[index]} at index: {index}")
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
    test = StressTestHashTable(preset_dynamic_objects, stress_number_of_items, Person, table_size=stress_number_of_items * 10)
    test.stress_test()
    


if __name__ == "__main__":
    main()
