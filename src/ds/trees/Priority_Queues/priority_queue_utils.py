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

# endregion


# region custom imports
from user_defined_types.generic_types import T
from utils.validation_utils import DsValidation
from utils.exceptions import *

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.priority_queue_adt import PriorityQueueADT

# endregion

class PriorityQueueUtils:
    """Util Methods for Priority Queues"""
    def __init__(self, priority_queue_obj) -> None:
        self.obj = priority_queue_obj

    def check_empty_pq(self):
        if self.obj.is_empty():
            raise DsUnderflowError("Error: Priority Queue is Empty.")

    def check_element_already_exists(self, element):
        if element in self.obj:
            raise DsDuplicationError("Error: Element already exists. Use 'Decrease Key()' to modify priority level.")

    def check_new_min_priority(self, new_priority: int, stored_priority: int):
        if new_priority > stored_priority:
            raise PriorityInvalidError("Error: Priority input must be lower than currently stored priority value.")

    def check_new_max_priority(self, new_priority: int, stored_priority: int):
        if new_priority < stored_priority:
            raise PriorityInvalidError("Error: Priority input must be higher than currently stored priority value.")

    def linear_scan_min(self, input_array) -> int:
        """Linear Scan: compare to all other elements in the array."""
        candidate = input_array.array[0]
        priority_index = 0
        for i in range(self.obj.size):
            kv_pair = self.obj._data.array[i]
            if self.obj._key(kv_pair) < self.obj._key(candidate):
                candidate = kv_pair
                priority_index = i
        return priority_index

    def linear_scan_max(self, input_array) -> int:
        """Linear Scan: compare to all other elements in the array."""
        candidate = input_array.array[0]
        priority_index = 0
        for i in range(self.obj.size):
            kv_pair = self.obj._data.array[i]
            if self.obj._key(kv_pair) > self.obj._key(candidate):
                candidate = kv_pair
                priority_index = i
        return priority_index

    def add_kv_pair_to_max_sorted_list(self, element, priority):
        """Automatically adds the item to the correct spot in the list."""
        kv_pair = (element, priority)
        found = False
        # traverse through items.
        for i in range(self.obj.size):
            current_element, current_priority = self.obj._data.array[i]
            if priority > current_priority:
                self.obj._data.insert(i, kv_pair)
                return
        # lowest priority case: -- if priority is the lowest - add to the end of the array
        self.obj._data.append(kv_pair)

    def compare_heap_nodes(self, child, parent) -> bool:
        """compares child and parent nodes - returns true or false
        - choose betwee min and max heap. can choose custom key if so desired
        """
        if self.obj.min_heap:
            return self.obj.key(child) < self.obj.key(parent)
        else:
            return self.obj.key(child) > self.obj.key(parent)

    def bubble_up_heap(self, index: int):
        """
        Compares Child and parent nodes, and swaps positions
        if current order violates heap-order property
        repeats process until heap-order is restored
        O(log n) - due to complete tree property.
        """
        # Step 1: compute parent index
        parent_index = (index - 1) // 2  
        # Step 2: loop through tree structure.
        while index > 0:
            # Step 3: define child and parent nodes
            child = self.obj._heap.array[index]
            parent = self.obj._heap.array[parent_index]

            # Step 4: Exit Condition: heap order is satisified
            if self.compare_heap_nodes(child, parent) == False:
                break

            # Step 5: (if heap order still violated) swap node positions.
            self.obj._heap.array[index], self.obj._heap.array[parent_index] = parent, child

            # Step 6: move up to next node
            index = parent_index    # move to parent index position.
            parent_index = (index - 1) // 2 # derive new parent index.

    def bubble_down_heap(self, index: int):
        """
        Compares a parent node to its children and swaps if the heap order is violated.
        """
        while index < self.obj.size:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            parent_index = index
            # If left child violates heap-order, set selected = left.
            if left_child_index < self.obj.size and self.compare_heap_nodes(self.obj._heap.array[left_child_index], self.obj._heap.array[parent_index]):
                parent_index = left_child_index
            # If right child violates heap-order more, set selected = right.
            if right_child_index < self.obj.size and self.compare_heap_nodes(self.obj._heap.array[right_child_index], self.obj._heap.array[parent_index]):
                parent_index = right_child_index
            # exit condition: heap order satisfied
            if parent_index == index:
                break
            # After comparing, if selected != index,
            # swap nodes - and move down tree.
            self.obj._heap.array[index], self.obj._heap.array[parent_index] = self.obj._heap.array[parent_index], self.obj._heap.array[index]
            index = parent_index
