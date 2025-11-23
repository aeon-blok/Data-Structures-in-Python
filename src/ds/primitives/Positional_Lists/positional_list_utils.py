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
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# endregion

# region custom imports

if TYPE_CHECKING:
    from user_defined_types.custom_types import T
    from adts.positional_list_adt import PositionalListADT, iNode, iPosition


from utils.exceptions import *


class PositionalListUtils:
    def __init__(self, pl_obj: "PositionalListADT[T]") -> None:
        self.obj = pl_obj


    def str_positional_list(self, sep: str = " ->> "):
        """Displays all the content of the linked list as a string."""
        seperator = sep
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        class_name = self.obj.__class__.__qualname__

        if self.obj.first() is None:
            return f"[{class_name}][{datatype}][{total_nodes}]"

        infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(self.positional_list_traversal())} (T)"
        return infostring

    def repr_positional_list(self):
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"

    def positional_list_traversal(self):
        """traverses the nodes and returns a string via generator"""
        current_pos = self.obj.first()
        while current_pos is not None:
            if current_pos is not None:
                yield current_pos.element
            current_pos = self.obj.after(current_pos)

    def validate_position(self, position: "iPosition"):
        """Ensures that the position is the correct type, is linked to a list, is referenced by the correct list and more..."""
        from adts.positional_list_adt import iPosition
        if not isinstance(position, iPosition):
            raise DsTypeError(f"Error: Invalid Type: Expected {iPosition.__name__}, Got: {type(position)}")
        if position.container is not self.obj:
            raise NodeOwnershipError(f"Error: Position does not belong to this Positional list.")
        if position.node.next is None and position.node.prev is None:
            raise NodeDeletedError(f"Error: Position was deleted and is no longer valid (or linked to the Positional list.)")
        return position.node

    def check_not_sentinel(self, node: "iNode[T]"):
        """checks if the reference node is the begining or end of the list"""
        return None if node in (self.obj._header, self.obj._trailer) else node
    
    def relink_nodes(self, node: "iNode[T]"):
        """
        Relinks a node between two other nodes. -- helper function for position list (with sentinels)
        Core insertion logic reused by all public insert methods.
        Step 1: Destructure Node Input
        Step 2: Link neighbour pointers to new node
        Step 3: Link new node to neighbours
        """
        new_node = node
        previous_node = node.prev
        next_node = node.next

        # link neighbours
        previous_node.next = new_node
        next_node.prev = new_node
        # link new node to neighbours
        new_node.next = next_node
        new_node.prev = previous_node

        return new_node
