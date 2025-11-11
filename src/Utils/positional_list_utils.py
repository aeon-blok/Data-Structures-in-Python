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
    from utils.custom_types import T

from adts.positional_list_adt import PositionalListADT, iNode, iPosition
from ds.primitives.Positional_Lists.position import Position, PNode


# endregion
def positional_list_traversal(pl_obj):
    """traverses the nodes and returns a string via generator"""
    current_pos = pl_obj.first()
    while current_pos is not None:
        yield current_pos.element
        current_pos = pl_obj.after(current_pos)


def validate_position(pl_obj, position):
    """Ensures that the position is the correct type, is linked to a list, is referenced by the correct list and more..."""
    if not isinstance(position, iPosition):
        raise TypeError(f"Error: Invalid Type: Expected {iPosition.__name__}, Got: {type(position)}")
    if position.container is not pl_obj:
        raise ValueError(f"Error: Position: {position} does not belong to this Positional list.")
    if position.node.next is None and position.node.prev is None:
        raise ValueError(f"Error: Position was deleted and is no longer valid (or linked to the Positional list.)")
    return position.node

def make_position(pl_obj, node):
    """
    Creates a position for a node object (a position is a proxy controller for a node)
    Sentinel nodes are not exposed.
    """
    # empty list guard clause
    if node is pl_obj._header or node is pl_obj._trailer:
        return None
    # can lazy import if needed
    position_obj = Position(node, container=pl_obj)
    return position_obj

def insert_between(pl_obj, element, previous_node, next_node):
    """
    Inserts a node between two other nodes. helper function for position list (with sentinels)
    Core insertion logic reused by all public insert methods.
    Step 1: Create New Node Object
    Step 2: Link neighbour pointers to new node
    Step 3: Link new node to neighbours
    Step 4: Update Tracker
    Step 5: Create New Position Object (from new node.)
    """
    # initialize node object - can lazy import if you want to fix method call
    new_node = PNode(element, previous_node, next_node)
    # link neighbours
    previous_node.next = new_node
    next_node.prev = new_node
    # link new node to neighbours
    new_node.next = next_node
    new_node.prev = previous_node
    pl_obj._total_nodes += 1  # update tracker
    # method that creates a position object
    return make_position(pl_obj, new_node)
