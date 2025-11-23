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
from user_defined_types.generic_types import T
from utils.exceptions import *

from adts.deque_adt import DequeADT

# endregion


class DequeUtils:
    """"""
    def __init__(self, deque_obj) -> None:
        self.obj = deque_obj

    def check_empty_deque(self):
        """if the deque is empty raises an error."""
        if self.obj.is_empty():
            raise DsUnderflowError(f"Error: The Deque is empty.")

    def add_first_element(self, element):
        """Empty deque Case: is a special case. Once it has elements, the normal circular movement applies."""
        self.obj._buffer.array[self.obj._front] = element
        self.obj._deque_size = 1 # we dont decrement for the empty deque.

    def add_front_element(self, element): 
        """
        Main Case: Decrement front before inserting -- front now points to the new element.
        why? because front always points to the first element, if we add an element at the front...
        """
        self.obj._front = (self.obj._front - 1) % self.obj._capacity
        self.obj._buffer.array[self.obj._front] = element # assign value
        self.obj._deque_size += 1   # update tracker

    def add_rear_element(self, element):
        """
        Adds an element to the rear of a Deque.
        increments rear index
        """
        rear = (self.obj._front + self.obj._deque_size) % self.obj._capacity
        self.obj._buffer.array[rear] = element
        self.obj._deque_size += 1

    def remove_front_element(self):
        """
        removes an element from the front of the deque
        increments front index
        """
        old_value = self.obj._buffer.array[self.obj._front]

        # derference front index
        if self.obj.datatype in (object, ctypes.py_object):
            self.obj._buffer.array[self.obj._front] = None

        self.obj._front = (self.obj._front + 1) % self.obj._capacity
        self.obj._deque_size -= 1   # decrement size tracker
        return old_value

    def remove_rear_element(self):
        """
        removes an element from the rear of the deque
        decrements rear index (derived)
        """
        rear = (self.obj._front + self.obj._deque_size - 1) % self.obj._capacity
        old_value = self.obj._buffer.array[rear]

        # derference front index
        if self.obj.datatype in (object, ctypes.py_object):
            self.obj._buffer.array[rear] = None

        self.obj._deque_size -= 1  # decrement size tracker
        return old_value
