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
    TYPE_CHECKING
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# endregion

# region custom imports
from utils.custom_types import T

# endregion


# region SLL

def validate_node(sll_obj, node, node_type):
    if node is None:
        raise ValueError("Error: Reference Node Object cannot be None ")
    if not isinstance(node, node_type):
        raise TypeError(f"Error: Invalid Type: Expected {node_type.__name__}, Got: {type(node)}")
    if not node.is_linked:
        raise ValueError(f"Error: Reference Node: {node} was deleted and is no longer valid (or linked to the list.)")
    if node.list_owner is not sll_obj:
        raise ValueError(f"Error: Reference Node: {node} does not belong to this linked list.")

def empty_list_exception(sll_obj):
    if sll_obj.is_empty():
        raise ValueError(f"Error: The Linked list is empty. Total Nodes: {sll_obj._total_nodes}")


# endregion
