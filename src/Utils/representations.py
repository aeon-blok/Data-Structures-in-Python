from typing import TYPE_CHECKING

# region custom imports
from utils.positional_list_utils import positional_list_traversal
if TYPE_CHECKING:
    from adts.sequence_adt import SequenceADT
    from adts.linked_list_adt import LinkedListADT
    from adts.positional_list_adt import PositionalListADT

    from utils.custom_types import T

# endregion


# where we add console visualizations for the different data structure types - usually use these in __str__ or __repr__ or a utility function.

# region arrays
def str_array(array_obj: "SequenceADT[T]"):
    """a list of strings representing all the elements in the array"""
    items = ", ".join(str(array_obj.array[i]) for i in range(array_obj.size))
    return f"[{array_obj.__class__.__qualname__}][{array_obj.datatype.__name__}][{array_obj.size}/{array_obj.capacity}][{items}]"

def repr_array(array_obj: "SequenceADT[T]"):
    """array __repr__ - for devs"""
    class_address = f"<{array_obj.__class__.__qualname__} object at {hex(id(array_obj))}>"
    data_type = f"Type: {array_obj.datatype.__name__}"
    storage = f"Capacity: {array_obj.size}/{array_obj.capacity}"
    array_type = f"Array Type: {'Static' if array_obj.is_static == True else 'Dynamic'}"
    return f"{class_address}, {data_type}, {storage}, {array_type}"

def str_view(view_obj: "SequenceADT[T]"):
    """ __str__ for array views (similar to slices in python without the copying)"""
    items = ", ".join(str(view_obj[i]) for i in range(view_obj._length))
    return f"[{view_obj.__class__.__qualname__}][{view_obj.datatype.__name__}][{view_obj._length}][{items}]"


def repr_view(view_obj: "SequenceADT[T]"):
    """ __repr__ for array views (like slices)"""
    class_address = f"<{view_obj.__class__.__qualname__} object at {hex(id(view_obj))}>"
    items = ", ".join(str(view_obj[i]) for i in range(view_obj._length))
    return f"{class_address}, Type: {view_obj.datatype.__name__}, Total Elements: {view_obj._length}"


# endregion

# region linked lists

def str_ll_node(ll_node_obj: "LinkedListADT[T]"):
    node_element = f"{ll_node_obj.element}"
    return f"{node_element}"


def repr_sll_node(sll_node_obj: "LinkedListADT[T]"):
    class_address = f"<{sll_node_obj.__class__.__qualname__} object at {hex(id(sll_node_obj))}>"
    next_pointer = f"Next: {str(sll_node_obj.next)}"
    node_element = f"Node: {sll_node_obj.element}"
    linked = f"in list?: {sll_node_obj.is_linked}"
    owner = f"Owner: {repr(sll_node_obj.list_owner)}"
    return f"{class_address}, {node_element}, {next_pointer}, {linked}, {owner}"


def repr_dll_node(dll_node_obj: "LinkedListADT[T]"):
    class_address = (
        f"<{dll_node_obj.__class__.__qualname__} object at {hex(id(dll_node_obj))}>"
    )
    prev_pointer = f"Prev: {str(dll_node_obj.prev)}"
    next_pointer = f"Next: {str(dll_node_obj.next)}"
    node_element = f"Node: {dll_node_obj.element}"
    linked = f"in list?: {dll_node_obj.is_linked}"
    owner = f"Owner: {repr(dll_node_obj.list_owner)}"
    return f"{class_address}, {node_element}, {prev_pointer}, {next_pointer}, {linked}, {owner}"


def str_ll(ll_obj: "LinkedListADT[T]", sep: str = " ->> "):
    """Displays all the content of the linked list as a string."""
    seperator = sep
    datatype = ll_obj.datatype.__name__
    total_nodes = ll_obj.total_nodes
    class_name = ll_obj.__class__.__qualname__

    if ll_obj._head is None:
        return f"[{class_name}][{datatype}][{total_nodes}]"

    def _simple_traversal():
        """traverses the nodes and returns a string via generator"""
        current_node = ll_obj._head
        while current_node:
            yield str(current_node._element)
            current_node = current_node.next
            # exit condition for DCLL
            if current_node is ll_obj._head:
                break

    infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(_simple_traversal())} (T)"
    return infostring


def repr_ll(ll_obj: "LinkedListADT[T]"):
    """Displays the memory address and other useful info"""
    class_address = f"<{ll_obj.__class__.__qualname__} object at {hex(id(ll_obj))}>"
    datatype = ll_obj.datatype.__name__
    total_nodes = ll_obj.total_nodes
    return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"


# endregion

# region Positional Lists

def repr_p_node(dll_node_obj: "PositionalListADT[T]"):
    """"""
    class_address = (f"<{dll_node_obj.__class__.__qualname__} object at {hex(id(dll_node_obj))}>")
    prev_pointer = f"Prev: {str(dll_node_obj.prev)}"
    next_pointer = f"Next: {str(dll_node_obj.next)}"
    node_element = f"Node: {dll_node_obj.element}"
    return f"{class_address}, {node_element}, {prev_pointer}, {next_pointer}"

def repr_position(pl_obj):
    """"""
    class_address = (f"<{pl_obj.__class__.__qualname__} object at {hex(id(pl_obj))}>")
    node_element = f"Node: {pl_obj.element}"
    owner = f"Owner: {repr(pl_obj.container)}"
    return f"{class_address}, {node_element}, {owner}"


def str_positional_list(pl_obj, sep: str = " ->> "):
    """Displays all the content of the linked list as a string."""
    seperator = sep
    datatype = pl_obj.datatype.__name__
    total_nodes = pl_obj.total_nodes
    class_name = pl_obj.__class__.__qualname__

    if pl_obj.first() is None:
        return f"[{class_name}][{datatype}][{total_nodes}]"

    infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(positional_list_traversal(pl_obj))} (T)"
    return infostring


def repr_positional_list(pl_obj):
    """Displays the memory address and other useful info"""
    class_address = f"<{pl_obj.__class__.__qualname__} object at {hex(id(pl_obj))}>"
    datatype = pl_obj.datatype.__name__
    total_nodes = pl_obj.total_nodes
    return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"


# endregion

# stacks


# queues


# deques


# heaps


# Maps


# Trees


# Graphs
