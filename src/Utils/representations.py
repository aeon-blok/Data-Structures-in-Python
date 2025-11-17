from typing import TYPE_CHECKING

# region custom imports
if TYPE_CHECKING:
    from adts.sequence_adt import SequenceADT
    from adts.linked_list_adt import LinkedListADT
    from adts.positional_list_adt import PositionalListADT
    from utils.custom_types import T

from ds.primitives.Positional_Lists.positional_list_utils import PositionalListUtils
from ds.primitives.Linked_Lists.linked_list_utils import LinkedListUtils
from utils.helpers import Ansi
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
        self._ansi = Ansi()
        self._top_marker = self._ansi.color(f"(Top)", Ansi.GREEN)

    def str_ll_stack(self) -> str:
        """Stack __str__ representation"""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_nodes = self.obj.total_nodes

        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []{self._top_marker}"

        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_nodes}]: [{', '.join(elements)}]{self._top_marker}"

    def repr_ll_stack(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.total_nodes
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"

class ArrayStackRepr:
    """Array Stack Representation in the console"""
    def __init__(self, stack_obj) -> None:
        self.obj = stack_obj
        self._ansi = Ansi()
        self._top_marker = self._ansi.color(f"(Top)", Ansi.GREEN)

    def str_array_stack(self) -> str:
        """Stack __str__ representation"""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_elements = self.obj.size
        capacity = self.obj._array.capacity
        array_size = self.obj._array.size
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []{self._top_marker}"
        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_elements}] ({array_size}/{capacity}): [{', '.join(elements)}]{self._top_marker}"

    def repr_array_stack(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        size = self.obj.size
        return f"{class_address}, Type: {datatype},"

    def str_min_max_avg_stack(self) -> str:
        """representation for the min max stack."""
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_elements = self.obj.size
        min_element = self.obj.min
        max_element = self.obj.max
        avg_element = self.obj.average
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []{self._top_marker}"
        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_elements}]: {', '.join(elements)}{self._top_marker}, Min: {min_element}, Max: {max_element}, Avg: {avg_element:.2f}"

    def repr_min_max_avg_stack(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_elements = self.obj.size
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_elements}"


# queues
class llQueueRepr:
    """Linked list queue representation """
    def __init__(self, queue_obj) -> None:
        self.obj = queue_obj
        self._ansi = Ansi()
        self._front_marker = self._ansi.color(f"(front)", Ansi.GREEN)
        self._rear_marker = self._ansi.color(f"(rear)", Ansi.GREEN)

    def str_ll_queue(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        total_nodes = self.obj.size
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"
        elements = (str(element) for element in self.obj)
        return f"[{class_name}][{datatype}][{total_nodes}]: {self._front_marker}[{', '.join(elements)}]{self._rear_marker}"

    def repr_ll_queue(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        total_nodes = self.obj.size
        front_node = self.obj.front
        return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}, Front: {front_node}"

class CircArrayQueueRepr:
    """Linked list queue representation"""

    def __init__(self, queue_obj) -> None:
        self.obj = queue_obj
        self._ansi = Ansi()

    def str_circ_array_queue(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        queue_size = self.obj.queue_size

        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"
        
        front = self.obj.front
        rear = self.obj.rear

        def _element_generator(color=Ansi.GREEN):
            """colors the front and rear in a specified color"""
            for i in range(queue_size):
                index = (self.obj._front + i) % self.obj._capacity
                value = self.obj._buffer.array[index]

                if value in (front, rear):
                    yield self._ansi.color(f"{value}", color)
                else:
                    yield str(value)

        return f"[{class_name}][{datatype}][{queue_size}]: [F][{', '.join(_element_generator())}][R]"

    def repr_circ_array_queue(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        queue_size = self.obj.queue_size
        front = self.obj.front
        rear = self.obj.rear
        return f"{class_address}, Type: {datatype}, Size: {queue_size}, Front: {front} Rear: {rear}"


# deques
class CircDequeRepr:
    def __init__(self, deque_obj) -> None:
        self.obj = deque_obj
        self._ansi = Ansi()

    def str_circ_deque(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        deque_size = self.obj.deque_size
        capacity = self.obj._capacity

        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"

        front = self.obj.front
        rear = self.obj.rear

        def _element_generator():
            """colors the front and rear in a specified color"""
            for i in range(deque_size):
                index = (self.obj._front + i) % self.obj._capacity
                value = self.obj._buffer.array[index]

                if value == front:
                    yield self._ansi.color(f"{value}", Ansi.GREEN)
                elif value == rear:
                    yield self._ansi.color(f"{value}", Ansi.YELLOW)
                else:
                    yield str(value)

        return f"[{class_name}][{datatype}][{deque_size}/{capacity}]: [F][{', '.join(_element_generator())}][R]"

    def repr_circ_deque(self) -> str:
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        deque_size = self.obj.deque_size
        capacity = self.obj._capacity
        if self.obj.is_empty():
            return f"{class_address}, Type: {datatype}, Size: {deque_size}/{capacity}"

        front = self.obj.front
        rear = self.obj.rear
        return f"{class_address}, Type: {datatype}, Size: {deque_size}/{capacity}, Front: {front} Rear: {rear}"

class LlDequeRepr:
    def __init__(self, deque_obj) -> None:
        self.obj = deque_obj
        self._ansi = Ansi()

    def dll_str_deque(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        deque_size = self.obj._dll.total_nodes

        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"

        front = self.obj.front
        rear = self.obj.rear

        def _element_generator():
            """colors the front and rear in a specified color"""
            current_node = self.obj._dll.head
            while current_node:
                element = current_node.element
                if element == front:
                    yield self._ansi.color(f"{element}", Ansi.GREEN)
                elif element == rear:
                    yield self._ansi.color(f"{element}", Ansi.YELLOW)
                else:
                    yield str(current_node.element)
                current_node = current_node.next    # traverse

        return f"[{class_name}][{datatype}][{deque_size}]: [F][{', '.join(_element_generator())}][R]"

    def dll_repr_deque(self):
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        deque_size = self.obj._dll.total_nodes
        if self.obj.is_empty():
            return f"{class_address}, Type: {datatype}, Total Nodes: {deque_size}"

        front = self.obj.front
        rear = self.obj.rear

        return f"{class_address}, Type: {datatype}, Total Nodes: {deque_size}, Front: {front} Rear: {rear}"


# priority queues
class PQueueRepr:
    def __init__(self, priority_queue_obj) -> None:
        self.obj = priority_queue_obj
        self._ansi = Ansi()

    def str_simple_pq(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        size = self.obj.size
        capacity = self.obj.capacity

        # empty case:
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"

        priority_element = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)

        def _generate_items():
            for i in range(self.obj.size):
                kv_pair = self.obj._data.array[i]
                element, priority = kv_pair
                # color priority element
                if element == self.obj.priority:
                    yield self._ansi.color(f"{element}: (p{priority})", Ansi.GREEN)
                else:
                    yield f"{element}: (p{priority})"

        return f"[{class_name}][{datatype}][{size}/{capacity}]: [{', '.join(_generate_items())}]"

    def repr_simple_pq(self):
        """Displays the memory address and other useful info"""
        class_address = f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        datatype = self.obj.datatype.__name__
        size = self.obj.size
        capacity = self.obj.capacity

        if self.obj.is_empty():
            return f"{class_address}, Type: {datatype}, Size: {size}/{capacity}"

        priority_element = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)
        return f"{class_address}, Type: {datatype}, Size: {size}/{capacity}, Priority Element: {priority_element}"


# heaps
class BinaryHeapRepr:
    def __init__(self, priority_queue_obj) -> None:
        self.obj = priority_queue_obj
        self._ansi = Ansi()

    def str_heap(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        size = self.obj.size
        capacity = self.obj.capacity

        # empty case:
        if self.obj.is_empty():
            return f"[{class_name}][{datatype}]: []"

        priority_element = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)

        def _generate_items():
            for i in range(self.obj.size):
                kv_pair = self.obj._heap.array[i]
                element, priority = kv_pair
                # color priority element
                if element == self.obj.priority:
                    yield self._ansi.color(f"{element}: (p{priority})", Ansi.GREEN)
                else:
                    yield f"{element}: (p{priority})"

        return f"[{class_name}][{datatype}][{size}/{capacity}]: [{', '.join(_generate_items())}]"

    def repr_heap(self):
        """Displays the memory address and other useful info"""
        class_address = (
            f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>"
        )
        datatype = self.obj.datatype.__name__
        size = self.obj.size
        capacity = self.obj.capacity
        heap_type = self.obj.heap_type

        if self.obj.is_empty():
            return f"{class_address}, Type: {datatype}, Heap Type: {heap_type}, Size: {size}/{capacity}"

        priority_element = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)
        return f"{class_address}, Type: {datatype}, Heap Type: {heap_type}, Size: {size}/{capacity}, Priority Element: {priority_element}"


# Maps


# Trees
class TreeNodeRepr:
    def __init__(self, tree_node_obj) -> None:
        self.obj = tree_node_obj
        

    def repr_tnode(self):
        """Displays the memory address and other useful info"""
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        node_status = self.obj.alive
        return f"{class_address}, Type: {datatype}, Node Data: {self.obj.value}, Node Alive?: {node_status}"

    def str_tnode(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        return f"({class_name}: {datatype}) {self.obj.value}"

class GenTreeRepr:
    def __init__(self, tree_obj) -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    def repr_gen_tree(self):
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        total_elements =  f"Total Nodes: {len(self.obj)}"
        return f"{class_address}, Type: {datatype}, {total_elements}"

    def str_gen_tree(self):
        """
        Traverses the Tree via stack
        adds connector symbols in front of each node value, depending on whether it is the last child "└─" or one of many "├─",
        every node adds either " " if parent is last child (no vertical bar needed) or "| " if parent is not last child (vertical bar continues)
        the node & its display symbols are appended to a list for the final string output.
        """
        total_tree_nodes = len(self.obj)
        tree_height = self.obj.height(self.obj.root)
        if self.obj.root is None:
            return f"< 🌳 empty tree>"

        hierarchy = []
        tree = [(self.obj.root, "", True)]  # (node, prefix, is_last)

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print 🌲
            if node is self.obj.root:
                indicator = "🌲:"
            # decides what connector symbol appears before the node value when printing the tree.
            else: 
                indicator = "" if prefix == "" else ("└─" if is_last else "├─")

            # add to final string output
            hierarchy.append(f"{prefix}{indicator}{str(node.value)}") 

            # Build prefix for children - Vertical bars "│" are inherited from ancestors that are not last children
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterates over the node’s children in reverse. (left to right) --- enumerate gives index i for calculating new prefix.
            for i, child in enumerate(reversed(node.children)):
                last_child = (i==0)
                # Update ancestor flags: current node's is_last boolean affects all its children
                tree.append((child, new_prefix, last_child))
        # final string:
        node_structure = "\n".join(hierarchy)
        title = self._ansi.color(f"Tree: Depth First Search (DFS):", Ansi.GREEN)

        return f"\n{title}\nTotal Nodes: {total_tree_nodes}, Tree Height: {tree_height}\n{node_structure}\n"


class BinaryNodeRepr:
    def __init__(self, node_obj) -> None:
        self.obj = node_obj

    def str_binary_node(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        return f"({class_name}: {datatype}) {self.obj.element}"

    def repr_binary_node(self):
        """Displays the memory address and other useful info"""
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        node_status = self.obj.alive
        left_child = self.obj.left
        right_child = self.obj.right
        return f"{class_address}, Type: {datatype}, Node Data: {self.obj.value}, Children: L: {left_child} R: {right_child} Node Alive?: {node_status}"


class BinaryTreeRepr:
    def __init__(self, tree_obj) -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    def str_binary_tree(self):
        total_tree_nodes = len(self.obj)
        tree_height = self.obj.height()
        if self.obj.root is None:
            return f"< 🌳 empty tree>"

        hierarchy = []
        tree = [(self.obj.root, "", True)]  # (node, prefix, is_last)

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print 🌲
            if node is self.obj.root:
                indicator = ""
            # decides what connector symbol appears before the node value when printing the tree.
            else:
                indicator = "" if prefix == "" else ("└─" if is_last else "├─")

            # add to final string output
            hierarchy.append(f"{prefix}{indicator}{str(node.element)}")

            # Build prefix for children - Vertical bars "│" are inherited from ancestors that are not last children
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterates over the node’s children in reverse. (left to right) --- enumerate gives index i for calculating new prefix.
            # ! pack into tuples and then into a list to iterate over. (for binary trees)
            children = []
            if node.right is not None:
                children.append((node.right, True))
            if node.left is not None:
                children.append((node.left, False))

            for child, last_flag in children:
                # Update ancestor flags: current node's is_last boolean affects all its children
                if child is not None:
                    tree.append((child, new_prefix, last_flag))

        # final string:
        node_structure = "\n".join(hierarchy)
        title = self._ansi.color(f"Tree: Depth First Search (DFS):🌲", Ansi.GREEN)

        return f"\n{title}\nTotal Nodes: {total_tree_nodes}, Tree Height: {tree_height}\n{node_structure}\n"

    def repr_binary_tree(self):
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        total_elements =  f"Total Nodes: {len(self.obj)}"
        return f"{class_address}, Type: {datatype}, {total_elements}"


# Graphs
