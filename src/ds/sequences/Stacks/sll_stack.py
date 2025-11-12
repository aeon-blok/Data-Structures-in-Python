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
from utils.representations import LlStackRepr

from utils.exceptions import *

from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from adts.stack_adt import StackADT

from ds.primitives.Linked_Lists.ll_nodes import Sll_Node
from ds.sequences.Stacks.stack_utils import StackUtils
from ds.primitives.Linked_Lists.sll import LinkedList


# endregion

""" 
A stack in a sense is a specific adaptation of a sequence. 
Where its insertion and removal and retrieval of elements in confined to a single index [0] - the Top.
Users never see or manipulate nodes directly.
"""

class Stackll(CollectionADT[T], StackADT[T], Generic[T]):
    def __init__(self, datatype: type) -> None:
        self._top: Optional["iNode[T]"] = None
        # self._total_nodes: int = 0
        self._datatype = datatype

        # Composed Objects
        self._linkedlist = LinkedList[T](self._datatype)
        self._utils = StackUtils(self)
        self._validators = DsValidation()
        self._desc = LlStackRepr(self)

    @property
    def datatype(self):
        return self._datatype
    @property
    def top(self):
        return None if self.is_empty() else self._linkedlist.head.element
    @property
    def linkedlist(self):
        return self._linkedlist
    @property
    def total_nodes(self):
        return self._linkedlist.total_nodes

    # ------------ Utility ------------
    def __str__(self) -> str:
        return self._desc.__str__()

    def __repr__(self) -> str:
        return self._desc.__repr__()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        return len(self._linkedlist)

    def __contains__(self, element: T) -> bool:
        return self._linkedlist.__contains__(element)

    def clear(self):
        self._linkedlist.clear()

    def is_empty(self) -> bool:
        return self._linkedlist.is_empty()

    def __iter__(self) -> Generator[T, None, None]:
        return self._linkedlist.__iter__()

    # ----- Canonical ADT Operations -----
    def push(self, element):
        """Insert a new element value to the top of the stack"""
        self._linkedlist.insert_head(element)

    def pop(self):
        """remove and return the top element of the stack."""
        return self._linkedlist.delete_head()

    def peek(self):
        """return the top element value without deletion"""
        return self._linkedlist.head.element




# main  ---- Client Facing Code

def main():
    print("=== Stackll Test Suite ===\n")

    stack = Stackll(int)
    print(stack)
    print(repr(stack))
    print(f"Testing is_empty? {stack.is_empty()}\n")

    # Push Operations
    print("Testing Push Operations:")
    for val in [10, 20, 30, 40]:
        print(f"Pushing {val}...")
        stack.push(val)
        print(stack)

    # Peek Top Element
    try:
        print(f"\nPeek Top Element: {stack.peek()}")
        print(f"Current Top Property: {stack.top}")
    except Exception as e:
        print(f"Error: {e}")

    # Pop Operations
    print("\nTesting Pop Operations:")
    try:
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
    except Exception as e:
        print(f"Error: {e}")

    # Mixed Push/Pop
    print("\nTesting Mixed Push/Pop:")
    stack.push(50)
    print(stack)
    try:
        popped = stack.pop()
        print(f"Popped: {popped}")
        print(stack)
    except Exception as e:
        print(f"Error: {e}")

    # Iteration
    print("\nTesting Iteration:")
    for val in [60, 70, 80]:
        stack.push(val)
    for i, item in enumerate(stack):
        print(f"Iterated over {i}: {item}")

    # Contains
    print("\nTesting __contains__:")
    print(f"Is 70 in stack? {70 in stack}")
    print(f"Is 999 in stack? {999 in stack}")

    # Length
    print("\nTesting __len__:")
    print(f"Stack length: {len(stack)}")

    # Clear
    print("\nTesting clear():")
    stack.clear()
    print(stack)
    print(f"Testing is_empty? {stack.is_empty()}")

    # Pop/Peek from empty stack
    print("\nTesting errors on empty stack:")
    try:
        stack.pop()
    except Exception as e:
        print(f"Caught error as expected: {e}")

    try:
        stack.peek()
    except Exception as e:
        print(f"Caught error as expected: {e}")

    print("\n=== Stackll Test Complete ===")


if __name__ == "__main__":
    main()
