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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.representations import CircDequeRepr
from utils.helpers import RandomClass

from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from adts.deque_adt import DequeADT
from ds.primitives.arrays.dynamic_array import VectorArray, VectorView
from ds.sequences.Deques.deque_utils import DequeUtils


# endregion



"""
Circular Array Deque:
"""


class CircularArrayDeque(DequeADT[T], CollectionADT[T], Generic[T]):
    """
    Circular Array Deque: with dynamic resizing
    One important thing to note that is seen with queues and deques, is the concept of the logical front, rather than the exact 0 index.
    in a Deque the front can be any number and often is, because of the circular cycling caused by adding to and from both the front and back of the deque.
    """
    def __init__(self, datatype: type, capacity: int = 10) -> None:
        self._datatype = datatype
        self._capacity = max(4, capacity)
        self._front: int = 0
        self._deque_size: int = 0

        # composed objects
        self._buffer = VectorArray(self._capacity, self._datatype, is_static=True)
        self._utils = DequeUtils(self)
        self._validators = DsValidation()
        self._desc = CircDequeRepr(self)

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----

    @property
    def datatype(self):
        return self._datatype
    
    @property
    def front(self):
        self._utils.check_empty_deque()
        return self._buffer.array[self._front]
    
    @property
    def rear(self):
        self._utils.check_empty_deque()
        rear_index = (self._front + self._deque_size - 1) % self._capacity
        return self._buffer.array[rear_index]
    
    @property
    def deque_size(self):
        return self._deque_size
    
    # ----- Utilities -----
    def __str__(self) -> str:
        return self._desc.str_circ_deque()
    
    def __repr__(self) -> str:
        return self._desc.repr_circ_deque()

    # ----- Meta Collection ADT Operations -----

    def __len__(self) -> int:
        return self._deque_size
    
    def __contains__(self, value: T) -> bool:
        for i in range(self._deque_size):
            index = (self._front + i) % self._capacity
            if self._buffer.array[index] == value:
                return True
        return False
    
    def clear(self) -> None:
        # reset trackers
        self._front = 0
        self._deque_size = 0
        self._buffer = VectorArray(self._capacity, self._datatype, is_static=True)

    def is_empty(self) -> bool:
        return self._deque_size == 0
    
    def __iter__(self):
        """iterate over a circular deque starting from the logical front"""
        for i in range(self._deque_size):
            yield self._buffer.array[(self._front + i) % self._capacity]

    # ----- Mutator ADT Operations -----
    def _resize_deque(self):
        """resizes deque - at the moment only grows. copies all the elements from the old deque to the new deque."""
        old_buffer = self._buffer
        old_capacity = self._capacity

        new_capacity = old_capacity * 2
        new_buffer = VectorArray(new_capacity, self._datatype, is_static=True)

        # copy to new array.
        for i in range(self._deque_size):
            new_buffer.array[i] = old_buffer.array[(self._front + i) % self._capacity]
    
        # reset trackers. (deque size doesnt change)
        self._buffer = new_buffer
        self._front = 0
        self._capacity = new_capacity
  
    def add_front(self, element):
        """add an element to the front of the deque"""
        self._validators.enforce_type(element, self.datatype)

        # resize Case:
        if self._deque_size == self._capacity:
            self._resize_deque()

        # Empty deque Case: 
        if self.is_empty():
            self._utils.add_first_element(element)
        else:
            # Main Case (decrement front index and add value)
            self._utils.add_front_element(element)

    def add_rear(self, element):
        """add an element to the rear of the deque"""
        self._validators.enforce_type(element, self.datatype)
        # resize Case:
        if self._deque_size == self._capacity:
            self._resize_deque()

        # Empty deque Case: 
        if self.is_empty():
            self._utils.add_first_element(element)
        else:
        # Main Case: Increment rear after inserting -- rear always points to last...
            self._utils.add_rear_element(element)

    def remove_front(self):
        """ remove an element from the front of the deque"""
        self._utils.check_empty_deque()
        return self._utils.remove_front_element()

    def remove_rear(self):
        """remove an element from the rear of the deque"""
        # Main Case: Remove rear: decrement rear (wrap modulo).
        self._utils.check_empty_deque()
        return self._utils.remove_rear_element()


    

# main ---- client facing code ----


def main():
    print("=== CircularArrayDeque Test Suite ===\n")

    # Create a deque of integers
    deque = CircularArrayDeque(int, capacity=4)
    print("Initial deque:", deque)
    print(repr(deque))
    
    # --- Test 1: Add rear ---
    print("\n-- Adding to rear --")
    for val in [10, 20, 30]:
        deque.add_rear(val)
        print(f"Added {val} to rear:", deque)

    # --- Test 2: Add front ---
    print("\n-- Adding to front --")
    deque.add_front(5)
    print("Added 5 to front:", deque)
    
    # --- Test 3: Wraparound ---
    print("\n-- Trigger wraparound --")
    deque.remove_front()  # remove 5, front moves
    deque.add_rear(40)
    print("Removed front and added 40 to rear (wraparound):", deque)

    # --- Test 4: Dynamic resizing ---
    print("\n-- Trigger dynamic resizing --")
    deque.add_rear(50)  # triggers resize
    print("Added 50 to rear (resize should happen):", deque)
    
    # --- Test 5: Remove front and rear ---
    print("\n-- Removing front and rear --")
    front_val = deque.remove_front()
    rear_val = deque.remove_rear()
    print(f"Removed front: {front_val}, rear: {rear_val}")
    print("Deque after removals:", deque)

    # --- Test 6: Iteration ---
    print("\n-- Iteration test --")
    print("Deque elements:", list(deque))

    print(f"\n Testing peek front and rear")
    print(f"Front: {deque.front}, Rear: {deque.rear}")

    # --- Test 7: __contains__ check ---
    print("\n-- __contains__ test --")
    print("Contains 20?", 20 in deque)
    print("Contains 999?", 999 in deque)

    # --- Test 8: Clear test ---
    print("\n-- Clear test --")
    deque.clear()
    print("Deque after clear:", deque)
    print("Is empty?", deque.is_empty())

    # --- Test 9: Type safety ---
    print("\n-- Type safety test --")
    try:
        deque.add_rear(RandomClass("EEEOOOT"))
    except TypeError as e:
        print("Caught type error as expected:", e)

    # --- Test 10: Underflow errors ---
    print("\n-- Underflow test --")
    try:
        deque.remove_front()
    except DsUnderflowError as e:
        print("Caught underflow error as expected:", e)

    try:
        deque.remove_rear()
    except DsUnderflowError as e:
        print("Caught underflow error as expected:", e)

    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()