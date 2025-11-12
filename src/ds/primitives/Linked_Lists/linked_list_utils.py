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
    Type,
    TYPE_CHECKING
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# endregion

# region custom imports
from utils.exceptions import *


if TYPE_CHECKING:
    from utils.custom_types import T
    from adts.linked_list_adt import LinkedListADT, iNode
# endregion




class LinkedListUtils:
    def __init__(self, ll_obj: "LinkedListADT[T]") -> None:
        """helper utilities for linked list data structures"""
        self.obj = ll_obj

    def validate_node(self, node: "iNode[T]", node_type: Type["iNode[T]"]):
        """Checks the node reference input"""
        if node is None:
            raise NodeEmptyError("Error: Reference Node Object cannot be None ")
        if not isinstance(node, node_type):
            raise NodeTypeError(f"Error: Invalid Type: Expected [{node_type.__name__}], Got: [{type(node)}]")
        if not node.is_linked:
            raise NodeDeletedError(f"Error: Reference Node [{node}] was deleted and is no longer valid (or linked to the list.)")
        if node.list_owner is not self.obj:
            raise NodeOwnershipError(f"Error: Reference Node: [{node}] does not belong to this linked list.")

    def assert_list_not_empty(self):
        """checks if the linked list is empty"""
        if self.obj.is_empty():
            raise DsUnderflowError(f"Error: The Linked list is empty. Total Nodes: {self.obj._total_nodes}")

    def check_node_exists(self, node: "iNode[T]"):
        """check if a node exists."""
        if node is None:
            raise NodeEmptyError("Node cannot be None, please give a valid Node.")

    def check_node_after_exists(self, node: "iNode[T]"):
        """checks if there is a node after the specified node"""
        if not node.next:
            raise NodeExistenceError("Error: No node exists after the specified node...")

    def check_node_before_exists(self, node: "iNode[T]"):
        """Checks there is a node before the reference node. Useful for insertions and deletions"""
        if not node.prev:
            raise NodeExistenceError("No Node exists before the specified node...")

    # region SLL

    def find_sll_node_before_reference(self, ref_node: "iNode[T]"):
        """traverses the singly linked list to 1 node before the reference node"""
        current_node = self.obj._head
        while current_node and current_node.next != ref_node:
            current_node = current_node.next
        return current_node

    def assert_sll_reference_node_exists(self, current_node: "iNode[T]", ref_node: "iNode[T]"):
        """Checks if the reference node exists. - if None - its the tail. (used when traversing a Singly Linked List)"""
        if current_node is None:
            raise NodeNotFoundError(f"Error: Node ({ref_node}) was not found in the list.")

    # endregion

    # region DCLL

    def traverse_dcll_nodes(self):
        """generator that traverses a dcll and returns nodes (lazily)"""
        current_node = self.obj._head

        # empty list Case:
        if not current_node:
            return

        while current_node:
            yield current_node
            current_node = current_node.next
            if current_node is self.obj._head:
                break

    # endregion












# def validate_node(ll_obj: "LinkedListADT[T]", node: "iNode[T]", node_type: Type["iNode[T]"]):
#     """Checks the node reference input"""
#     if node is None:
#         raise ValueError("Error: Reference Node Object cannot be None ")
#     if not isinstance(node, node_type):
#         raise TypeError(f"Error: Invalid Type: Expected {node_type.__name__}, Got: {type(node)}")
#     if not node.is_linked:
#         raise ValueError(f"Error: Reference Node: {node} was deleted and is no longer valid (or linked to the list.)")
#     if node.list_owner is not ll_obj:
#         raise ValueError(f"Error: Reference Node: {node} does not belong to this linked list.")


# def assert_list_not_empty(ll_obj: "LinkedListADT[T]"):
#     """checks if the linked list is empty"""
#     if ll_obj.is_empty():
#         raise IndexError(f"Error: The Linked list is empty. Total Nodes: {ll_obj._total_nodes}")


# def check_node_exists(node: "iNode[T]"):
#     """check if a node exists."""
#     if node is None:
#         raise IndexError("Node cannot be None, please give a valid Node.")


# def check_node_after_exists(node: "iNode[T]"):
#     """checks if there is a node after the specified node"""
#     if not node.next:
#         raise IndexError("Error: No node exists after the specified node...")


# def check_node_before_exists(node: "iNode[T]"):
#     """Checks there is a node before the reference node. Useful for insertions and deletions"""
#     if not node.prev:
#         raise IndexError("No Node exists before the specified node...")


# def find_node_before_reference(sll_obj: "LinkedListADT[T]", ref_node: "iNode[T]"):
#     """traverses the singly linked list to 1 node before the reference node"""
#     current_node = sll_obj._head
#     while current_node and current_node.next != ref_node:
#         current_node = current_node.next
#     return current_node


# def assert_reference_node_exists(current_node: "iNode[T]", ref_node: "iNode[T]"):
#     """Checks if the reference node exists. - if None - its the tail. (used when traversing a Singly Linked List)"""
#     if current_node is None:
#         raise IndexError(f"Error: Node {ref_node}: was not found in the list.")


# def traverse_dcll_nodes(ll_obj: "LinkedListADT[T]"):
#     """generator that traverses a dcll and returns nodes (lazily)"""
#     current_node = ll_obj._head

#     # empty list Case:
#     if not current_node:
#         return

#     while current_node:
#         yield current_node
#         current_node = current_node.next
#         if current_node is ll_obj._head:
#             break
