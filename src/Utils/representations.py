from typing import TYPE_CHECKING

# region custom imports
if TYPE_CHECKING:
    from adts.sequence_adt import SequenceADT
    from adts.linked_list_adt import LinkedListADT
    from adts.positional_list_adt import PositionalListADT
    from utils.custom_types import T

from ds.primitives.Positional_Lists.positional_list_utils import PositionalListUtils
from ds.primitives.Linked_Lists.linked_list_utils import LinkedListUtils
# endregion


# where we add console visualizations for the different data structure types - usually use these in __str__ or __repr__ or a utility function.

# region arrays
class ArrayRepr:
    def __init__(self, array_obj) -> None:
        self.obj = array_obj

    def str_array(self):
        """a list of strings representing all the elements in the array"""
        items = ", ".join(str(self.obj.array[i]) for i in range(self.obj.size))
        return f"[{self.obj.__class__.__qualname__}][{self.obj.datatype.__name__}][{self.obj.size}/{self.obj.capacity}][{items}]"

    def repr_array(self):
        """array __repr__ - for devs"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        data_type = f"Type: {self.obj.datatype.__name__}"
        storage = f"Capacity: {self.obj.size}/{self.obj.capacity}"
        array_type = f"Array Type: {'Static' if self.obj.is_static == True else 'Dynamic'}"
        return f"{class_address}, {data_type}, {storage}, {array_type}"

class ViewRepr:
    def __init__(self, view_obj) -> None:
        self.obj = view_obj
            
    def str_view(self):
        """ __str__ for array views (similar to slices in python without the copying)"""
        items = ", ".join(str(self.obj[i]) for i in range(self.obj._length))
        return f"[{self.obj.__class__.__qualname__}][{self.obj.datatype.__name__}][{self.obj._length}][{items}]"

    def repr_view(self):
        """ __repr__ for array views (like slices)"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        items = ", ".join(str(self.obj[i]) for i in range(self.obj._length))
        return f"{class_address}, Type: {self.obj.datatype.__name__}, Total Elements: {self.obj._length}"
# endregion


# region linked lists
class SllNodeRepr:
    """Representation for Singly Linked List NODE """
    def __init__(self, sll_node_obj) -> None:
        self.obj = sll_node_obj

    def str_ll_node(self):
        node_element = f"{self.obj.element}"
        return f"{node_element}"

    def repr_sll_node(self):
        class_address = (
            f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        )
        next_pointer = f"Next: {str(self.obj.next)}"
        node_element = f"Node: {self.obj.element}"
        linked = f"in list?: {self.obj.is_linked}"
        owner = f"Owner: {repr(self.obj.list_owner)}"
        return f"{class_address}, {node_element}, {next_pointer}, {linked}, {owner}"

class DllNodeRepr:
    """Representation for Doubly Linked List NODE"""
    def __init__(self, dll_node_obj) -> None:
        self.obj = dll_node_obj

    def str_ll_node(self):
        node_element = f"{self.obj.element}"
        return f"{node_element}"

    def repr_dll_node(self):
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        prev_pointer = f"Prev: {str(self.obj.prev)}"
        next_pointer = f"Next: {str(self.obj.next)}"
        node_element = f"Node: {self.obj.element}"
        linked = f"in list?: {self.obj.is_linked}"
        owner = f"Owner: {repr(self.obj.list_owner)}"
        return f"{class_address}, {node_element}, {prev_pointer}, {next_pointer}, {linked}, {owner}"

class LinkedListRepr:
    """Linked List Representation"""
    def __init__(self, ll_obj) -> None:
        self.obj = ll_obj

    def str_ll(self, sep: str = " ->> "):
        """Displays all the content of the linked list as a string."""
        seperator = sep
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        class_name = self.obj.__class__.__qualname__

        if self.obj._head is None:
            return f"[{class_name}][{datatype}][{total_nodes}]"

        def _simple_traversal():
            """traverses the nodes and returns a string via generator"""
            current_node = self.obj._head
            while current_node:
                yield str(current_node._element)
                current_node = current_node.next
                # exit condition for DCLL
                if current_node is self.obj._head:
                    break

        infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(_simple_traversal())} (T)"
        return infostring

    def repr_ll(self):
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"
# endregion


# region Positional Lists
class PNodeRepr:
    def __init__(self, node_obj) -> None:
        self.obj = node_obj
    """Represntations for the Position Nodes"""
    def repr_p_node(self):
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        prev_pointer = f"Prev: {str(self.obj.prev)}"
        next_pointer = f"Next: {str(self.obj.next)}"
        node_element = f"Node: {self.obj.element}"
        return f"{class_address}, {node_element}, {prev_pointer}, {next_pointer}"

class PositionRepr:    
    """Representations for the Position Object"""
    def __init__(self, pos_obj) -> None:
        self.obj = pos_obj

    def repr_position(self):
            """"""
            class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
            node_element = f"Node: {self.obj.element}"
            owner = f"Owner: {repr(self.obj.container)}"
            return f"{class_address}, {node_element}, {owner}"

class PlistRepr:
    """Representations for the actual position list itself."""
    def __init__(self, pl_obj) -> None:
        self.obj = pl_obj
        self._utils = PositionalListUtils(self.obj)

    def str_positional_list(self, sep: str = " ->> "):
        """Displays all the content of the linked list as a string."""
        seperator = sep
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        class_name = self.obj.__class__.__qualname__

        if self.obj.first() is None:
            return f"[{class_name}][{datatype}][{total_nodes}]"
        
        infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(self._utils.positional_list_traversal())} (T)"
        return infostring

    def repr_positional_list(self):
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"
# endregion


# stacks

class LlStackRepr:
    """Stack Representation in the console."""
    def __init__(self, stack_obj) -> None:
        self.obj = stack_obj

    def __str__(self) -> str:
        """Stack __str__ representation"""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_nodes = self.obj.total_nodes

        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: [Top][]"

        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_nodes}]: [Top][{', '.join(elements)}]"

    def __repr__(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes

        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"


class ArrayStackRepr:
    """Array Stack Representation in the console"""
    def __init__(self, stack_obj) -> None:
        self.obj = stack_obj

    def str_array_stack(self) -> str:
        """Stack __str__ representation"""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_elements = self.obj.size
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: [](Top)"
        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_elements}]: [{', '.join(elements)}](Top)"

    def repr_array_stack(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"

    def str_min_max_stack(self) -> str:
        """representation for the min max stack."""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_elements = self.obj.size
        min_element = self.obj.min
        max_element = self.obj.max
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: [](Top)"
        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_elements}]: {', '.join(elements)}(Top), Min: {min_element}, Max: {max_element}"

    def repr_min_max_stack(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_elements = self.obj.size
        min_element = self.obj.min
        max_element = self.obj.max
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_elements}, Min: {min_element}, Max: {max_element}"


# queues


# deques


# heaps


# Maps


# Trees


# Graphs
