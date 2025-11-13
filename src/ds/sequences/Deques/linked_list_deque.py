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
from utils.representations import LlDequeRepr
from utils.helpers import RandomClass
from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from adts.deque_adt import DequeADT

from ds.primitives.Linked_Lists.ll_nodes import Dll_Node
from ds.primitives.Linked_Lists.dll import DoublyLinkedList
from ds.sequences.Deques.deque_utils import DequeUtils


# endregion


class DllDeque(DequeADT[T], CollectionADT[T], Generic[T]):
    """Deque implementation via Doubly Linked List."""
    def __init__(self, datatype: type) -> None:
        self._datatype = datatype

        # composed object:
        self._dll = DoublyLinkedList(self._datatype)
        self._utils = DequeUtils(self)
        self._desc = LlDequeRepr(self)

    @property
    def datatype(self):
        return self._datatype

    @property
    def front(self):
        head = self._dll.head.element
        return head

    @property
    def rear(self):
        return self._dll.tail.element

    # ----- Meta Collection ADT Operations -----

    def __len__(self) -> int:
        return self._dll.total_nodes

    def __contains__(self, value: T) -> bool:
        return self._dll.__contains__(value)

    def is_empty(self) -> bool:
        return self._dll.is_empty()

    def clear(self) -> None:
        return self._dll.clear()

    def __iter__(self):
        return self._dll.__iter__()

    # ------------ Utilities ----------

    def __str__(self) -> str:
        return self._desc.dll_str_deque()

    def __repr__(self) -> str:
        return self._desc.dll_repr_deque()

    # ----- Canonical ADT Operations -----
    def add_front(self, element):
        self._dll.insert_head(element)

    def add_rear(self, element):
        self._dll.insert_tail(element)

    def remove_front(self):
        element: T = self._dll.delete_head()
        return element

    def remove_rear(self):
        element: T = self._dll.delete_tail()
        return element


# Main ---- Client Facing Code ----

def main():
    # --- Integer deque ---
    int_deque = DllDeque(int)

    # Add multiple elements
    for i in range(10):
        int_deque.add_rear(i * 10)  # 0, 10, 20, ..., 90

    for i in range(5):
        int_deque.add_front(-i * 10)  # 0, -10, -20, -30, -40 at front

    print("Integer deque with more items:", int_deque)
    print("Front element:", int_deque.front)
    print("Rear element:", int_deque.rear)
    print(repr(int_deque))

    # Remove some elements
    removed_front = int_deque.remove_front()
    removed_rear = int_deque.remove_rear()
    print(f"Removed front: {removed_front}")
    print(f"Removed rear: {removed_rear}")
    print("Deque now:", int_deque)
    print(repr(int_deque))

    # Iteration over deque
    print("Iterating over integer deque:")
    for elem in int_deque:
        print(elem, end=", ")
    print("\nLength:", len(int_deque))

    # --- String deque ---
    str_deque = DllDeque(str)
    fruits = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    for fruit in fruits:
        str_deque.add_rear(fruit)

    print("String deque with multiple items:", str_deque)
    print(repr(str_deque))

    # Remove front and rear
    f_str = str_deque.remove_front()
    r_str = str_deque.remove_rear()
    print(f"Removed front (str): {f_str}")
    print(f"Removed rear (str): {r_str}")
    print("String deque now:", str_deque)

    # --- Mixed type error check ---
    try:
        str_deque.add_front(RandomClass("5436543534534"))  # wrong type
    except TypeError as e:
        print("Caught type error (mixed type):", e)

    # --- Empty deque check ---
    empty_deque = DllDeque(float)
    print("Empty float deque:", empty_deque)
    print("Is empty?", empty_deque.is_empty())

    # --- Empty deque tests ---
    empty_deque = DllDeque(int)

    print("Testing empty deque operations:")

    # Access front
    try:
        print("Front:", empty_deque.front)
    except Exception as e:
        print("Access front on empty deque raised:", e)

    # Access rear
    try:
        print("Rear:", empty_deque.rear)
    except Exception as e:
        print("Access rear on empty deque raised:", e)

    # Remove front
    try:
        removed = empty_deque.remove_front()
        print("Removed front:", removed)
    except Exception as e:
        print("remove_front on empty deque raised:", e)

    # Remove rear
    try:
        removed = empty_deque.remove_rear()
        print("Removed rear:", removed)
    except Exception as e:
        print("remove_rear on empty deque raised:", e)

    print(repr(empty_deque))


if __name__ == "__main__":
    main()
