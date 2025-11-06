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
For this implementation we will handle collisions via Open Addressing & Linear Probing (via Tombstones)
"""


# Custom Types
T = TypeVar("T")


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
    def keys(self) -> Optional["VectorArray"]:
        """Return a set of all the keys in the hash table"""
        pass

    @abstractmethod
    def values(self) -> Optional["VectorArray"]:
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


class VectorArray(Generic[T]):
    """Dynamic Array — automatically resizes as elements are added."""

    def __init__(
        self, capacity: int, datatype: type, datatype_map: dict = CTYPES_DATATYPES
    ) -> None:

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
        string_capacity = f"Capacity: {self.size}/{self.capacity}"
        return f"[{string_datatype}][{items}][{string_capacity}]"

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


class OAHashTable(MapADT[T]):
    """Hash Table Data Structure with Probing / double hashing & Tombstones (Open Addressing)"""

    def __init__(
        self,
        datatype: type,
        capacity: int = 10,
        max_load_factor: float = 0.6,
        resize_factor: int = 2,
        probes_threshold: float = 0.15,
        tombstones_threshold: float = 0.15,
        average_probes_limit: float = 4,
        probing_technique: Literal["linear", "quadratic", "double hashing"] = "double hashing",
    ):
        self.min_capacity = max(4, self._find_next_prime_number(capacity))
        self.capacity = self._find_next_prime_number(capacity)
        self.enforce_type = datatype
        # initialize table.
        self.table = VectorArray(self.capacity, object)
        for i in range(self.capacity):
            self.table.array[i] = None

        # core attributes
        self.total_elements = 0   # tracks the number of kv pairs in the table
        self.max_load_factor = max_load_factor # prevents the table from exceeding this capacity
        self.current_load_factor = self._calculate_load_factor()  # log attribute - displays current load factor
        self.resize_factor = resize_factor
        # unique tombstone class. used as a tombstone marker
        self.tombstone = type("Tombstone", (), {"__repr__": lambda self: "🪦", "__str__": lambda self: "🪦"})()

        # MAD compression function parameters
        self.prime = self._find_next_prime_number(self.capacity)
        self.scale = random.randint(2, self.prime-1)  
        self.shift = random.randint(2, self.prime-1)

        # Probing Technique
        self.probing_technique = probing_technique

        # trackers
        self.current_collisions = 0
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_tombstones = 0  
        self.current_probes = 0
        self.total_probes = 0
        self.total_probe_operations = 0
        self.collisions_threshold: float = 0.15
        self.tombstones_threshold: float = tombstones_threshold
        self.probe_threshold: float = probes_threshold
        self._collisions_ratio = self.current_collisions / self.capacity
        self._tombstones_ratio = self.current_tombstones / self.capacity
        self._probe_ratio = self.current_probes / self.capacity
        self._average_probe_length: float = 0.0
        self.average_probe_limit: float = average_probes_limit

    @property
    def collisions_ratio(self):
        return self.current_collisions / self.capacity

    @property
    def tombstones_ratio(self):
        return self.current_tombstones / self.capacity

    @property
    def probe_ratio(self):
        return self.current_probes / self.capacity

    @property
    def average_probe_length(self):
        if self.total_elements == 0:
            return 0.0
        return self.total_probes / self.total_elements

    @average_probe_length.setter
    def average_probe_length(self, value):
        self._average_probe_length = value

    # ----- Utility -----

    def _create_stats(self, stats_only: bool=False, no_color: bool=True):
        """Creates formatted stats output for use with __str__ or other custom functions..."""
        DIM = "\033[2m"  # dim/faint
        BLUE = "\033[1;36m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        MAGENTA = "\033[35m"
        RESET = "\033[0m"
        MUTED_BLUE = f"{DIM}{BLUE}"

        def color(text: str, color_code: str):
            return f"{color_code}{text}{RESET}"

        # table items strings
        items = self.items()
        table_items = f"{str(', '.join(f'{k}: {v}' for k, v in items))}"

        # Load Factor:
        load_factor_emoji = "🏋️" or "🚚"
        color_load_factor = color(f"{self.current_load_factor:.2f}", GREEN) if self.current_load_factor < self.max_load_factor  else color(f"{self.current_load_factor:.2f}", RED)
        load_factor_string = f"{load_factor_emoji} : {color_load_factor}"
        load_factor_nocolor = f"{load_factor_emoji} : {self.current_load_factor:.2f}"
        # capacity string
        capacity_string = f"[{self.total_elements}/{self.capacity}]"
        # datatype string
        datatype_string = f"[{self.enforce_type.__name__}]"
        # collisions string
        collisions_emojis = f"💥" or "⚠️"
        color_collisions = color(f"{self.current_collisions}", GREEN) if self.collisions_ratio < self.collisions_threshold - 0.05 else color(f"{self.current_collisions}", RED)
        total_collisions_string = f"{collisions_emojis} : {color_collisions}"
        total_coll_nocolor = f"{collisions_emojis} : {self.current_collisions}"

        # tombstone string
        tombstone_emojis = f"🪦" or "💀"
        color_tombstones = color(f"{self.current_tombstones}",GREEN) if  self.tombstones_ratio < self.tombstones_threshold - 0.05 else color(f"{self.current_tombstones}",RED)
        tombstone_string = f"{tombstone_emojis}  : {color_tombstones}"
        tombstones_nocolor = f"{tombstone_emojis}  : {self.current_tombstones}"

        # rehash string
        rehash_emoji = f"♻️" or "⚙️" or "🔧"
        rehashes_string = f"{rehash_emoji}  : {self.total_rehashes}"

        # current probe length
        probe_emoji = f"🔍"
        color_probes = color(f"{self.current_probes}" , GREEN) if self.probe_ratio < self.probe_threshold - 0.05 else color(f"{self.current_probes}", RED)
        probes_string = f"{probe_emoji} : {color_probes}"
        probes_nocolor = f"{probe_emoji} : {self.current_probes}"

        # average probe length
        average_probe_emoji = f"Avg 🔍"
        color_avg_probe = color(f"{self.average_probe_length:.2f}" , GREEN) if self.average_probe_length < 3 else color(f"{self.average_probe_length:.2f}", RED)
        avg_probes_string = f"{average_probe_emoji} : {color_avg_probe}"
        avg_probes_nocolor = f"{average_probe_emoji} : {self.average_probe_length:.2f}"

        # compact statistics about the metrics for the hash table.
        repr_object = f"<{self.__class__.__qualname__} object at {hex(id(self))}>"
        if not no_color:
            stats = f"{repr_object}: {datatype_string}{capacity_string}[{load_factor_string}, {probes_string}, {tombstone_string}, {total_collisions_string}, {rehashes_string}, {avg_probes_string}]"
        else:
            stats = f"{repr_object}: {datatype_string}{capacity_string}[{load_factor_nocolor}, {probes_nocolor}, {tombstones_nocolor}, {total_coll_nocolor}, {rehashes_string}, {avg_probes_nocolor}]"

        # final composite
        infostring = f"{datatype_string}{capacity_string}{{{{{table_items}}}}}"

        return stats if stats_only else infostring

    def _display_table(self, columns: int = 12, cell_width: int = 15, row_padding: int = 3):
        """Table visualization - with tombstone markers included!"""

        table = self.table.array
        table_container = []

        # table creation.
        columns = columns
        cell_width = cell_width
        row_seperator = "-" * (columns * (cell_width + row_padding))

        # traverse every item in table
        # - if there is an item add the index number as text to the slot. - otherwise add the tombstone marker or []
        for idx, item in enumerate(table):
            if item == self.tombstone:
                table_container.append("🪦")
            elif item is None:
                table_container.append("")
            else:
                table_container.append(f"i: {idx}")
                # table_container.append("💬")

        # rows logic ---
        table_size = len(table_container)

        # title
        print(row_seperator)
        hashtable_type_string = (f"(Type: [{self.enforce_type.__name__}])")
        title = f"Hash Table Visualization: {hashtable_type_string}"
        stats = self._create_stats(stats_only=True, no_color=True)
        print(title.center(len(row_seperator)))
        print(row_seperator)
        print(stats.center(len(row_seperator)))
        print(row_seperator)

        # create rows
        for i in range(0, len(table_container), columns):
            row = table_container[i:i+columns]
            row_display = [str(item).center(cell_width) for item in row]
            print(" | ".join(row_display))
            print(row_seperator)

    # ----- Probing Function -----
    def linear_probing_function(self, index) -> int:
        """traverses through hashtable looking for empty slot"""
        return (index + 1) % self.capacity

    def quadratic_probing_function(self, start_index, probe_count) -> int:
        """quadratic probing function."""
        linear_term = 1  # linear term - stops quad from missing slots
        quadratic_term = 3  # quadratic term - provides spread to probes

        return (start_index + linear_term * probe_count + quadratic_term * (probe_count**2)) % self.capacity

    def double_hashing(self, key, probe_count) -> int:
        """Double Hashing - uses second hash as a step size - better spread probing function"""
        first_index = self._hash_function(key)
        second_step_size_index = self._second_hash_function(key)
        return (first_index + probe_count * second_step_size_index) % self.capacity

    def select_probing_function(self, key, index, start_index, probe_count) -> int:
        """Selects between different probing functions (quadratic, linear, double hashing)"""
        if self.probing_technique == "linear":
            new_index = self.linear_probing_function(index)
        elif self.probing_technique == "quadratic":
            new_index = self.quadratic_probing_function(start_index, probe_count)
        elif self.probing_technique == "double hashing":
            new_index = self.double_hashing(key, probe_count)
        else:
            raise ValueError(f"Error: {self.probing_technique}: Invalid Probing Technique entered. Please select from valid options.")
        return new_index

    # ----- Validation Checks & Errors -----
    def _enforce_type(self, value):
        """type enforcement - checks that the value matches the prescribed datatype."""
        if not isinstance(value, self.enforce_type):
            raise TypeError(f"Error: Invalid Type: Expected: {self.enforce_type.__name__} Got: {type(value)}")

    # ----- Python Built in Overrides -----
    def __str__(self) -> str:
        """prints whenever the item is printed in the console"""
        stats = self._create_stats(stats_only=False)
        return stats

    def __repr__(self) -> str:
        """prints dev info"""
        stats = self._create_stats(stats_only=True, no_color=False)
        return stats

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)

    def __delitem__(self, key):
        return self.remove(key)

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
        index = divide % self.capacity  # finally mod by table capacity
        return index

    def _second_hash_function(self, key):
        """creates a simple second hash function for step size for double hashing"""
        second_hash_code = self._cyclic_shift_hash_code(key)
        return 1 + (second_hash_code % (self.capacity - 1))

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

    def _hash_function(self, key: str):
        """Combines the hash code and compression function and returns an index value for a key."""
        poylnomial_hashcode = self._polynomial_hash_code(key)   # better for smaller tables like up to 1000 (0 collisions)
        cyclic_shift_hashcode = self._cyclic_shift_hash_code(key)   # better for huge tables like 10,000+ (1000 collisions)

        index = self._mad_compression_function(cyclic_shift_hashcode)
        return index

    # ----- Table Rehashing -----
    def _calculate_load_factor(self) -> float:
        """calculates the load factor of the current hashtable"""
        return self.total_elements / self.capacity

    def _rehash_table(self):
        """Rehashes table - copies items from an old table to a new table - and resets tracking counters"""
        start_time = time.perf_counter()

        # Store Old hash table
        old_capacity = self.capacity
        old_table = self.table.array

        # Set new capacity (normally * 2)
        new_capacity = self._find_next_prime_number(old_capacity * self.resize_factor)

        # initialize new table.
        new_table = VectorArray(new_capacity, object)
        for i in range(new_capacity):
            new_table.array[i] = None

        # reset trackers
        # new MAD modifiers
        self.prime = self._find_next_prime_number(new_capacity) # self.table_capacity * 300000000
        self.scale = random.randint(2, self.prime - 1)
        self.shift = random.randint(2, self.prime - 1)
        # reinitialize table with new size.
        self.table = new_table  
        self.capacity = new_capacity
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
        self.current_load_factor = self._calculate_load_factor()

    def _rehash_condition(self) -> bool:
        """will rehash the table if any 1 of these conditions is true."""
        if self.current_load_factor > self.max_load_factor:
            return True
        if self.probe_ratio > self.probe_threshold:
            return True
        if self.tombstones_ratio > self.tombstones_threshold:
            return True
        if self.average_probe_length > self.average_probe_limit:
            return True
        return False

    def _internal_put(self, key, value):
        """For use with the rehash functionality only -- does not use the rehash condition."""
        self._enforce_type(value)
        index = self._hash_function(key)    # calculate index
        start_index = index # set start index for probe function
        tombstone_start_index = None
        probe_count = 0

        # Probing Function: travel through the
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

            # moves to the next index on the table - This is the core of linear probing.
            index = self.select_probing_function(key, index, start_index, probe_count)

            # /Exit Condition: if we get back to where we started with no empty slot - the table is full
            if index == start_index:
                break

        # Default Condition: Add kv pair to index
        target_index = tombstone_start_index if tombstone_start_index is not None else index
        # equivalence check: if we replace a tombstone - decrement tombstones counter.
        if self.table.array[target_index] == self.tombstone:
            self.current_tombstones -= 1
        self.table.array[target_index] = (key, value)
        self.total_elements += 1
        self.current_load_factor = self._calculate_load_factor()
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

        self._enforce_type(value)   # type enforcement on the values entered into the hash table.

        # table rehash conditions
        if self._rehash_condition():
            self._rehash_table()

        index = self._hash_function(key)    # calculate index
        start_index = index # set start index for probe function
        tombstone_start_index = None
        probe_count = 0  # number of probes until key is found or insertion succeeds
        # Probing Function: travel through the table - ignoring None and tombstones. (only actual kv pairs)
        while self.table.array[index] is not None:
            probe_count += 1    # adds on keys and tombstones
            # logic for tombstone
            if self.table.array[index] == self.tombstone:
                if tombstone_start_index is None:   # only cache the first tombstone index we find...
                    tombstone_start_index = index
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

            # moves to the next index on the table - This is the core of linear probing.
            index = self.select_probing_function(key, index, start_index, probe_count)

            # Error/Exit Condition: if we get back to where we started with no empty slot - the table is full
            if index == start_index:
                raise RuntimeError(f"Error: Hash table is full.")

        # Default Condition: Add kv pair to index
        # defines the index as either the first tombstone that was found, or the current index.
        target_index: int = tombstone_start_index if tombstone_start_index is not None else index
        # equivalence check: if we replace a tombstone - decrement tombstones counter.
        if self.table.array[target_index] == self.tombstone:
            self.current_tombstones -= 1
        self.table.array[target_index] = (key, value)
        # updates trackers
        self.total_elements += 1
        self.current_load_factor = self._calculate_load_factor()
        self.current_probes = probe_count
        # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
        self.total_probes += self.current_probes
        self.total_probe_operations += 1    

    def get(self, key, default=None):
        """retrieves a key value pair from the hash table, with an optional default if the key is not found."""
        index = self._hash_function(key)
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

            # moves to the next slot in the table.
            index = self.select_probing_function(key, index, start_index, probe_count)

            # Exit Condition: if we have traversed the whole table and nothing found, break while loop and return default.
            if index == start_index:
                break

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
        if self._rehash_condition():
            self._rehash_table()

        index = self._hash_function(key)
        start_index = index
        probe_count = 0

        # find key at index. (skip None and Tombstone markers)
        while self.table.array[index] is not None: 
            probe_count += 1
            if self.table.array[index] != self.tombstone:
                k, v = self.table.array[index]
                # if the key matches - add tombstone marker to the table index specifically
                if k == key:
                    self.table.array[index] = self.tombstone
                    # update trackers.
                    self.total_elements -= 1
                    self.current_tombstones += 1
                    self.current_load_factor = self._calculate_load_factor()
                    return v

            # moves to the next index
            index = self.select_probing_function(key, index, start_index, probe_count)

            # Exit Condition: looped the whole way round....
            if index == start_index:
                break

        # update current probes metric for trackers
        self.current_probes = probe_count
        # adds the current probes for this operation to an aggregrated total used to calculate average probes per operation
        self.total_probes += self.current_probes
        self.total_probe_operations += 1

        # raise error
        raise KeyError(f"Error: Key: {key} not found...")

    def keys(self):
        """Return a set of all the keys in the hash table"""
        found = VectorArray(self.capacity, str)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                found.append(k)
        return found

    def values(self):
        """Return a set of all the values in the hash table"""
        found = VectorArray(self.capacity, self.enforce_type)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                found.append(v)
        return found

    def items(self):
        """Return a set of all the values in the hash table"""
        found = VectorArray(self.capacity, tuple)
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                found.append(slot)
        return found

    # ----- Meta Collection ADT Operations -----
    def __len__(self):
        """Returns the number of key-value pairs in the hash table"""
        return self.total_elements

    def contains(self, key):
        """Does the Hash table contain an item with the specified key?"""
        index = self._hash_function(key)
        start_index = index
        while self.table.array[index] is not None:
            if self.table.array[index] != self.tombstone:
                k, v = self.table.array[index]
                if k == key:
                    return True
            linear_probing = (index + 1) % self.capacity
            index = linear_probing

            if index == start_index:
                break

        return False

    def is_empty(self):
        return self.total_elements == 0

    def clear(self):
        """Resets the table to initial empty space with original capacity. resets all trackers also."""
        # reset trackers
        self.capacity = self.min_capacity
        # new MAD modifiers
        self.prime = self._find_next_prime_number(self.capacity)
        self.scale = random.randint(2, self.prime - 1)
        self.shift = random.randint(2, self.prime - 1)
        self.total_elements = 0  # reset item count
        self.current_tombstones = 0 # reset tombstones count
        self.current_collisions = 0   # reset collisions count
        self.total_rehashes = 0
        self.total_rehash_time = 0.0
        self.current_probes = 0
        self.current_load_factor = self._calculate_load_factor()    # reset load factor
        # update average probe metrics
        self.average_probe_length = 0.0
        self.total_probes = 0
        self.total_probe_operations = 0

        # reinitialize table.
        self.table = VectorArray(self.capacity, object)   
        for i in range(self.capacity):
            self.table.array[i] = None

    def __iter__(self):
        """The default iteration for a Map, is to generate a sequence (list) of all the keys in the map."""
        for slot in self.table.array:
            if slot is not None and slot != self.tombstone:
                k, v = slot
                yield k


# Main ---- Client Facing Code ------

# todo add the ability to change dynamic array to static (or just freeze shrink array) (low priority)


def main():

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

    input_values = [*input_data * 60]
    random.shuffle(input_values)

    # --- Initialize Hash Table ---
    hashtable = OAHashTable(Person, capacity=20, max_load_factor=0.6, probing_technique='double hashing')
    print("Created hash table:", hashtable)

    # testing put() logic
    for i, key in enumerate(input_values):
        hashtable.put(f"key_{i}", key)
        print(repr(hashtable))    # testing __str__

    # testing remove logic
    delete_items = list(hashtable.items())
    delete_subset = random.sample(delete_items, min(len(delete_items) // 5, 1000))
    for pair in delete_subset:
        k,v = pair
        hashtable.remove(k)
        print(repr(hashtable))

    # testing __getitem & __setitem__
    items = list(hashtable.items())
    subset = random.sample(items, min(5, len(items) // 10))
    for i, kv_pair in enumerate(subset):
        random_key = random.choice(hashtable.keys())
        k,v = kv_pair
        getitem = hashtable[random_key]
        print(f"Retrieving Item from Table: Got: {getitem}")
    for k, v in subset:
        random_value = random.choice(hashtable.values())
        hashtable[k] = random_value
        print(f"Updating Value: {hashtable[k]} with new value {random_value}... Expected: {random_value} Got: {hashtable[k]}")

    # iterating over items() keys() & values()
    # for k,v in hashtable.items():
    #     print(f"{k}={v}")
    # print(f", ".join(hashtable.keys()))
    # print(f", ".join(hashtable.values()))

    # test type safety:
    try:
        print(f"\nTesting Invalid type input: {wrong_type}")
        hashtable.put("wrong_type", wrong_type)
    except Exception as e:
        print(f"{e}")

    # test __contains__
    print(f"\nCheck if Invalid Type: {wrong_type}: Exists in the table currently?\nExpected: False, Got: {hashtable.contains('wrong_type')}\n")

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


    # test clear()
    print(f"Clearing Table: ")
    hashtable.clear()
    print(repr(hashtable))
    print(f"Total Elements in Hash Table Currently: {len(hashtable)}")


if __name__ == "__main__":
    main()
