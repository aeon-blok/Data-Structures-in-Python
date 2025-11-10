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


def validate_node(ll_obj, node, node_type):
    """Checks the node reference input"""
    if node is None:
        raise ValueError("Error: Reference Node Object cannot be None ")
    if not isinstance(node, node_type):
        raise TypeError(f"Error: Invalid Type: Expected {node_type.__name__}, Got: {type(node)}")
    if not node.is_linked:
        raise ValueError(f"Error: Reference Node: {node} was deleted and is no longer valid (or linked to the list.)")
    if node.list_owner is not ll_obj:
        raise ValueError(f"Error: Reference Node: {node} does not belong to this linked list.")

def assert_list_not_empty(ll_obj):
    """checks if the linked list is empty"""
    if ll_obj.is_empty():
        raise ValueError(f"Error: The Linked list is empty. Total Nodes: {ll_obj._total_nodes}")

def check_node_exists(node):
    """check if a node exists."""
    if node is None:
        raise ValueError("Node cannot be None, please give a valid Node.")

def check_node_after_exists(node):
    """checks if there is a node after the specified node"""
    if not node.next:
        raise IndexError("Error: No node exists after the specified node...")

def check_node_before_exists(node):
    """Checks there is a node before the reference node. Useful for insertions and deletions"""
    if not node.prev:
        raise IndexError("No Node exists before the specified node...")


# region SLL

def find_node_before_reference(sll_obj, ref_node):
    """traverses the singly linked list to 1 node before the reference node"""
    current_node = sll_obj._head
    while current_node and current_node.next != ref_node:
        current_node = current_node.next
    return current_node

def assert_reference_node_exists(current_node, ref_node):
    """Checks if the reference node exists. - if None - its the tail. (used when traversing a Singly Linked List)"""
    if current_node is None:
        raise ValueError(f"Error: Node {ref_node}: was not found in the list.")

# endregion
