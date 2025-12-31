# region standard lib
from types import UnionType
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
from faker import Faker

# endregion

# region custom imports
from user_defined_types.generic_types import (
    T,
    K,
    ValidDatatype,
    ValidIndex,
    TypeSafeElement,
    Index,
)
from user_defined_types.hashtable_types import (
    NormalizedFloat,
    LoadFactor,
    HashCodeType,
    CompressFuncType,
)
from user_defined_types.key_types import iKey, Key

from utils.constants import (
    MIN_HASHTABLE_CAPACITY,
    BUCKET_CAPACITY,
    HASHTABLE_RESIZE_FACTOR,
    DEFAULT_HASHTABLE_CAPACITY,
    MAX_LOAD_FACTOR,
)

from utils.validation_utils import DsValidation
from utils.representations import SkipListRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.sorted_map_adt import SortedMapADT

from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.primitives.Linked_Lists.ll_nodes import SkipNode, SkipNodeSentinel
from ds.primitives.Linked_Lists.linked_list_utils import LinkedListUtils

from user_defined_types.key_types import iKey, Key
from user_defined_types.generic_types import (
    ValidDatatype,
    TypeSafeElement,
    Index,
    PositiveNumber,
)

# endregion


class SkipList(SortedMapADT[T, K], CollectionADT[T]):
    """
    Level 0 of the skip list contains all keys, in sorted order
    A node appears in level i if its height > i
    Sentinel is the Head for every level
    """
    def __init__(self, datatype: type, capacity: int = 100) -> None:
        self._datatype = ValidDatatype(datatype)
        self._keytype: None | type = None

        # This should be roughly log 1/p(N) where N is the expected number of elements.
        # Memory vs. Speed: Increasing the number of levels consumes more memory but speeds up searches for very large datasets
        self.max_level = math.ceil(math.log2(capacity))
        self._level = 0
        self._probability: float = 0.5
        self._head = SkipNodeSentinel(self.max_level)
        self._tail = None
        self._size = 0

        # composed objects
        self._validators = DsValidation()
        self._desc = SkipListRepr(self)

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> Index:
        return self._size

    def clear(self) -> None:
        return super().clear()

    def is_empty(self) -> bool:
        return self._size == 0

    def __contains__(self, key) -> bool:
        """does the skip list contain the specified key."""
        pred = self._skip_list_search(key)
        node = pred[0].forward[0]
        return node is not None and node.key == key

    def __iter__(self):
        """returns all the keys in sorted order."""
        current = self._head.forward[0]
        while current is not None:
            yield current.key
            current = current.forward[0]

    # ----- Utility -----

    def _randomly_generate_height(self):
        """Simulates a coin toss and randomly generates a level for the new insertion"""
        level = 1
        while random.random() < self._probability and level < self.max_level:
            level += 1
        return level

    def __str__(self) -> str:
        return self._desc.str_skip_list()

    def __repr__(self) -> str:
        return self._desc.repr_skip_list()

    # ----- Canonical ADT Operations -----
    # ----- Accessors -----

    def entries(self) -> VectorArray[T] | None:
        """returns all the entries in the skip list as an array"""
        if self._size == 0: return None
        entries = VectorArray(self._size, tuple)
        current = self._head.forward[0]
        while current is not None:
            entries.append((current.key.value, current.element))
            current = current.forward[0]
        return entries

    def keys(self) -> VectorArray | None:
        """returns all the keys in the skip list as an array"""
        if self._size == 0: return None
        keys = VectorArray(self._size, self._keytype)
        current = self._head.forward[0]
        while current is not None:
            keys.append(current.key.value)
            current = current.forward[0]
        return keys

    def values(self) -> VectorArray | None:
        """returns all the elements in the skip list as an array"""
        if self._size == 0: return None
        elements = VectorArray(self._size, self._datatype)
        current = self._head.forward[0]
        while current is not None:
            elements.append(current.element)
            current = current.forward[0]
        return elements

    def find_min(self):
        """
        returns a tuple of the smallest key with its paired value.
        Note: the comparison is via the key, not the value itself.
        """
        if self._size == 0: return None

        node = self._head.forward[0]
        return (node.key.value, node.element)

    def find_max(self):
        """returns a tuple of the largest key, and its paired value"""
        if self._size == 0: return None
        return (self._tail.key.value, self._tail.element)

    def find_floor(self, key):
        """Returns the entry with the largest key <= the specified key"""

        pred = self._skip_list_search(key)

        # * key exists already:
        node = pred[0].forward[0]
        if node is not None and node.key == key:
            return (node.key.value, node.element)

        # * key does NOT exist:
        # the node just before where the key would be inserted at level 0
        predecessor = pred[0]

        # no key less than or equal to the specified key exists.
        if predecessor is self._head:
            return None

        # the predecessor is the largest key <= to the specified key.
        return (predecessor.key.value, predecessor.element)

    def find_ceiling(self, key):
        """returns the entry with the smallest key >= the specified key"""

        pred = self._skip_list_search(key)

        # This is either the exact key - if it exists, or the next largest key. aka successor
        node = pred[0].forward[0]

        # key doesnt exist:
        if node is None:
            return

        # key exists
        return (node.key.value, node.element)

    def predecessor(self, key):
        """convenience method - returns the predecessor of the specified key."""
        pred = self._skip_list_search(key)[0]
        if pred is self._head:
            return
        return (pred.key.value, pred.element)

    def successor(self, key) -> Tuple:
        """returns the successor of the specified key"""
        pred = self._skip_list_search(key)[0]
        succ = pred.forward[0]
        if succ is None:
            return
        return (succ.key.value, succ.element)

    def submap(self, start, stop) -> SortedMapADT[T, K]:
        return super().submap(start, stop)

    def rank(self, key) -> Index:
        return super().rank(key)

    # ----- Mutators -----

    def _skip_list_search(self, key):
        """
        this finds the predecessor for a specific key - at every level required
        this does not find the logical predecessor in the sorted map.
        it finds where to insert or remove a node
        """
        # container
        splice_points = [None] * self.max_level
        current = self._head

        # traverse skip list from Top level -> bottom level,
        # ignore sentinels, traverse along level linked list until we reach the specified key.
        for i in range(self._level-1, -1, -1):
            while current.forward[i] is not None and current.forward[i].key < key:
                current = current.forward[i]
            # add to the container - this node forward pointer will be rewired in insertion / deletion
            splice_points[i] = current

        return splice_points

    def put(self, key: K, value: T) -> T | None:
        """
        Inserts a kv pair into the skip list
        Also has Upsert functionality - if the key already exists, the value will be updated.
        Generates new skip list levels or express lanes for speedy traversal.
        Updates pointers for linked list structural validity
        """

        # * find insertion point for new kv pair
        insert_point = self._skip_list_search(key)

        # * Key already exists Case: Upsert: update element value, returns the old value...
        candidate = insert_point[0].forward[0]
        if candidate is not None and candidate.key == key:
            old_value = candidate.element
            candidate._element = TypeSafeElement(value, self._datatype)
            return old_value

        # * key doesnt exist Case:
        # randomly choose level to insert new node into (simulates a coin toss)
        height = self._randomly_generate_height()
        new_node = SkipNode(self._datatype, key, value, height)

        # * creates new skip list level if necessary (new express lane)
        if height > self._level:
            for i in range(self._level, height):
                # Head sentinel is the first node in the new level.
                insert_point[i] = self._head
            # update level tracker to match new height of skip list.
            self._level = height

        # * rewire pointers
        for i in range(height):
            new_node.forward[i] = insert_point[i].forward[i]
            insert_point[i].forward[i] = new_node
        # rewire pointers for level 0 (prev)
        if new_node.forward[0] is not None:
            new_node.forward[0].prev = new_node

        new_node.prev = insert_point[0]

        # update tail
        if new_node.forward[0] is None:
            self._tail = new_node

        # * increment total nodes tracker
        self._size += 1
        return

    def remove(self, key: K) -> T | None:
        """
        removes a kv pair from the skip list.
        also removes levels if they become empty.
        """

        # empty skip list check
        if self._size == 0:
            return

        # * retrieve deletion position
        deletion_point = self._skip_list_search(key)
        target = deletion_point[0].forward[0]

        # * key doesnt exist case:
        if target is None or target.key != key:
            return

        # * key exists in skip list:
        # rewire pointers
        for i in range(target.height):
            if deletion_point[i].forward[i] is target:
                deletion_point[i].forward[i] = target.forward[i]
        # fix level 0 pointers (prev)
        if target.forward[0] is not None:
            target.forward[0].prev = target.prev
        # update tail
        if target is self._tail:
            self._tail = target.prev

        # remove top levels if top levels become empty
        while self._level > 0 and self._head.forward[self._level-1] is None:
            self._level -= 1
        # decrement size tracker
        self._size -= 1

        # * 1 member list case:
        if self._size == 0:
            self._tail = self._head

        return target.element

    def get(self, key: K, default: T | None) -> T | None:
        """retrieves an element from the skip list"""
        pred_point = self._skip_list_search(key)
        candidate = pred_point[0].forward[0]
        if candidate is not None and candidate.key == key:
            return candidate.element
        return default


# Main --------------- Client Facing Code --------------------
