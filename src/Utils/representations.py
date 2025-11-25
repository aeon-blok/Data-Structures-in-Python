from typing import TYPE_CHECKING

# region custom imports
if TYPE_CHECKING:
    from adts.sequence_adt import SequenceADT
    from adts.linked_list_adt import LinkedListADT
    from adts.positional_list_adt import PositionalListADT
    from user_defined_types.generic_types import T

from ds.primitives.Positional_Lists.positional_list_utils import PositionalListUtils
from ds.primitives.Linked_Lists.linked_list_utils import LinkedListUtils
from utils.helpers import Ansi
from utils.constants import SLL_SEPERATOR
# endregion


# where we add console visualizations for the different data structure types - usually use these in __str__ or __repr__ or a utility function.

# todo create a base class for inheriting common attributes from.

class BaseRepr:
    """Holds all the common descriptors that are used by every representation."""
    def __init__(self, obj) -> None:
        self.obj = obj
        self._ansi = Ansi()

    @property
    def ds_class(self):
        return f"[{self.obj.__class__.__qualname__}]"

    @property
    def ds_memory_address(self):
        return f"[{self.obj.__class__.__qualname__}: {hex(id(self.obj))}]"

    @property
    def ds_datatype(self):
        return f"[Type: {self.obj.datatype.__name__}]"


class UntypedBaseRepr:
    """Holds all the common descriptors that are used by every representation."""

    def __init__(self, obj) -> None:
        self.obj = obj
        self._ansi = Ansi()

    @property
    def ds_class(self):
        return f"[{self.obj.__class__.__qualname__}]"

    @property
    def ds_memory_address(self):
        return f"[{self.obj.__class__.__qualname__}: {hex(id(self.obj))}]"


# region arrays
class ArrayRepr(BaseRepr):
    """Representation for arrays and base class for array type structures."""
    @property
    def items(self) -> str:
        return f"[{', '.join(str(self.obj.array[i]) for i in range(self.obj.size))}]"
    
    @property
    def storage(self) -> str:
        return f"[{self.obj.size}/{self.obj.capacity}]"
    
    @property
    def array_type(self) -> str:
        return f"[{'Static' if self.obj.is_static == True else 'Dynamic'}]"
    

    def str_array(self):
        """a list of strings representing all the elements in the array"""
        return f"{self.ds_class}{self.items}"

    def repr_array(self):
        """array __repr__ - for devs"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.array_type}"

class ViewRepr(ArrayRepr):
    """A View is similar to a Python slice, but doesnt copy items. works with the VectorArray."""

    @property
    def length(self) -> str:
        return f"[{self.obj._length}]"

    @property
    def view_items(self) -> str:
        return f"[{', '.join(str(self.obj[i]) for i in range(self.obj._length))}]"

    def str_view(self):
        """ __str__ for array views (similar to slices in python without the copying)"""
        return f"{self.ds_class}{self.view_items}"

    def repr_view(self):
        """ __repr__ for array views (like slices)"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.length}"
# endregion


# region linked lists
class SllNodeRepr(UntypedBaseRepr):
    """Representation for Singly Linked List NODE """

    @property
    def node_element(self) -> str:
        return f"{self.obj.element}"

    @property
    def next_pointer(self) -> str:
        return f"[Nxt: {str(self.obj.next)}]"

    @property
    def alive(self) -> str:
        return f"[Alive?: {self.obj.is_linked}]"

    @property
    def owner(self) -> str:
        instance = self.obj.list_owner
        owner_name = type(instance).__name__
        memory_address = hex(id(instance))
        return f"[Owner: {owner_name}: {memory_address}]"

    def str_ll_node(self):
        return f"{self.node_element}"

    def repr_sll_node(self):
        return f"{self.ds_memory_address}: {self.node_element}, {self.next_pointer}{self.alive}{self.owner}"

class DllNodeRepr(SllNodeRepr):
    """Representation for Doubly Linked List NODE"""

    @property
    def prev_pointer(self) -> str:
        return f"[Prv: {str(self.obj.prev)}]"

    def str_ll_node(self):
        return f"{self.node_element}"

    def repr_dll_node(self):
        return f"{self.ds_memory_address}: {self.node_element}, {self.next_pointer}{self.prev_pointer}{self.alive}{self.owner}"

class LinkedListRepr(BaseRepr):
    """Linked List Representation"""

    @property
    def total_nodes(self) -> str:
        return f"[{self.obj.total_nodes}]"

    @property
    def simple_traversal(self):
        """traverses the nodes and returns a string via generator"""
        current_node = self.obj._head
        while current_node:
            yield str(current_node._element)
            current_node = current_node.next
            # exit condition for DCLL
            if current_node is self.obj._head:
                break

    @property
    def head_symbol(self) -> str:
        return f"(H)"

    @property
    def tail_symbol(self) -> str:
        return f"(T)"

    def str_ll(self, sep: str = SLL_SEPERATOR):
        """Displays all the content of the linked list as a string."""
        # empty ll case:
        if self.obj._head is None: return f"{self.ds_class}{self.ds_datatype}{self.total_nodes}"
        infostring = f"{self.ds_class}{self.total_nodes}: {self.head_symbol} {sep.join(self.simple_traversal)} {self.tail_symbol}"
        return infostring

    def repr_ll(self):
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}"

# endregion


# region Positional Lists
class PNodeRepr(UntypedBaseRepr):
    """Representations for the Position Nodes"""

    @property
    def node_element(self):
        return f"{self.obj.node_element}"

    @property
    def prev_pointer(self) -> str:
        return f"[Prv: {str(self.obj.prev)}]"

    @property
    def next_pointer(self) -> str:
        return f"[Nxt: {str(self.obj.next)}]"

    def str_p_node(self):
        return f"{self.node_element}"

    def repr_p_node(self):
        return f"{self.ds_memory_address}: {self.node_element}, {self.next_pointer}{self.prev_pointer}"

class PositionRepr(UntypedBaseRepr):
    """Representations for the Position Object"""

    @property
    def owner(self) -> str:
        instance = self.obj.container
        owner_name = type(instance).__name__
        memory_address = hex(id(instance))
        return f"[Owner: {owner_name}: {memory_address}]"

    @property
    def position_element(self) -> str:
        return f"{self.obj.element}"

    def str_position(self):
        return f"{self.position_element}"

    def repr_position(self):
        return f"{self.ds_memory_address}: elem = {self.position_element}, {self.owner}"

class PlistRepr(BaseRepr):
    """Representations for the actual position list itself."""
    def __init__(self, obj) -> None:
        super().__init__(obj)
        self._utils = PositionalListUtils(self.obj)

    @property
    def total_nodes(self) -> str:
        return f"[{self.obj.total_nodes}]"

    @property
    def head_symbol(self) -> str:
        return f"(H)"

    @property
    def tail_symbol(self) -> str:
        return f"(T)"

    @property
    def head(self) -> str:
        position = self.obj.head
        position = None if position is None else self.obj.head.element
        return f"[Head = {position}]"

    @property
    def tail(self) -> str:
        position = self.obj.tail
        position = None if position is None else self.obj.tail.element
        return f"[Tail = {position}]"

    def str_positional_list(self, sep: str = SLL_SEPERATOR):
        """Displays all the content of the linked list as a string."""    
        if self.obj.first() is None: return f"{self.ds_class}{self.total_nodes}"
        infostring = f"{self.ds_class}{self.total_nodes}: {self.head_symbol} {sep.join(self._utils.positional_list_traversal())} {self.tail_symbol}"
        return infostring

    def repr_positional_list(self):
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.head}{self.tail}"

# endregion


# stacks
class LlStackRepr(LinkedListRepr):
    """Stack Representation in the console."""
    def __init__(self, obj) -> None:
        super().__init__(obj)

    @property
    def top_symbol(self) -> str:
        return self._ansi.color(f"(Top)", Ansi.GREEN)

    @property
    def top_element(self) -> str:
        top = self.obj.top
        color_top = self._ansi.color(f"{top}", Ansi.GREEN)
        return f"[Top={color_top}]"
    
    @property
    def elements(self) -> str:
        elements = (str(element) for element in self.obj)
        return f"[{', '.join(elements)}]"

    def str_ll_stack(self) -> str:
        """Stack __str__ representation"""
        if self.obj.is_empty(): return f"{self.ds_class}{self.total_nodes}: []"
        return f"{self.ds_class}{self.total_nodes}: {self.top_element}{self.elements}"

    def repr_ll_stack(self) -> str:
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.top_element}"

class ArrayStackRepr(ArrayRepr):
    """Array Stack Representation in the console"""

    @property
    def top_element(self) -> str:
        top = self.obj.data.array[self.obj.top]
        color_top = self._ansi.color(f"{top}", Ansi.GREEN)
        if self.obj.is_empty(): color_top = self._ansi.color(f"None", Ansi.GREEN)
        return f"[Top={color_top}]"

    @property
    def elements(self) -> str:
        elements = (str(element) for element in self.obj)
        elements_string = f"[{', '.join(elements)}]"
        return elements_string

    @property
    def storage(self) -> str:
        number_of_elems = self.obj.size
        array_capacity = self.obj.data.capacity
        return f"[{number_of_elems}/{array_capacity}]"

    def str_array_stack(self) -> str:
        """Stack __str__ representation"""
        if self.obj.is_empty(): return f"{self.ds_class}: []"
        return f"{self.ds_class}{self.storage}: {self.top_element}{self.elements}"

    def repr_array_stack(self) -> str:
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.top_element}"

class MinMaxStackRepr(ArrayStackRepr):
    """console visualization for the MinMaxAvg Stack"""

    @property
    def min(self)->str:
        min = self.obj.min
        color_min = self._ansi.color(f"{min}", Ansi.GREEN)
        if self.obj.is_empty():
            color_min = self._ansi.color(f"None", Ansi.GREEN)
        return f"[Min={color_min}]"

    @property
    def max(self)->str:
        max = self.obj.max
        color_max = self._ansi.color(f"{max}", Ansi.RED)
        if self.obj.is_empty():
            color_max = self._ansi.color(f"None", Ansi.RED)
        return f"[Max={color_max}]"

    @property
    def average(self)->str:
        avg = self.obj.average
        color_avg = self._ansi.color(f"{avg}", Ansi.YELLOW)
        return f"Avg={color_avg}]"

    @property
    def key(self)->str:
        """Potentially use later - fill with default and custom"""
        numeric = lambda x: x
        count = lambda x: len(x)
        lexographic = lambda x: len(x)
        key = "None"
        if self.obj.key is None and issubclass(self.obj.datatype, (int, float)):
            key = "Numeric"
        elif self.obj.key is None and issubclass(self.obj.datatype, (list, dict, set)):
            key = "Count Elements"
        elif self.obj.key is None and issubclass(self.obj.datatype, (str, tuple)):
            key = "Lexographic"
        elif self.obj.key is None and issubclass(self.obj.datatype, complex):
            key = "Complex Numeric"
        elif self.obj.key is not None:
            key = "Custom"
        return f"[Key={key}]"

    def str_min_max_avg_stack(self) -> str:
        """representation for the min max stack."""
        if self.obj.is_empty(): return f"{self.ds_class}{self.storage}: []"
        return f"{self.ds_class}{self.storage}: {self.elements}"

    def repr_min_max_avg_stack(self) -> str:
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.min}{self.max}{self.average}{self.key}"

# queues
class llQueueRepr(LinkedListRepr):
    """Linked list queue representation """

    @property
    def front_marker(self):
        return self._ansi.color(f"(F)", Ansi.GREEN)

    @property
    def rear_marker(self):
        return self._ansi.color(f"(R)", Ansi.GREEN)

    @property
    def elements(self):
        def generate_elements():
            current_node = self.obj.linkedlist.head
            while current_node:
                value = str(current_node.element)
                if current_node is self.obj.linkedlist.head:
                    value = self._ansi.color(value, Ansi.GREEN)
                if current_node is self.obj.linkedlist.tail:
                    value = self._ansi.color(value, Ansi.GREEN)
                yield value
                current_node = current_node.next

        elements_string = f"[{', '.join(generate_elements())}]"
        return elements_string

    @property
    def front_element(self) -> str:
        front = self.obj.front
        front_color = self._ansi.color(f"{front}", Ansi.GREEN)
        return f"[Front={front_color}]"

    @property
    def rear_element(self) -> str:
        rear = self.obj.rear
        rear_color = self._ansi.color(f"{rear}", Ansi.GREEN)
        return f"[Rear={rear_color}]"

    @property
    def total_nodes(self) -> str:
        total_nodes = self.obj.queue_size
        return f"[{total_nodes}]"

    def str_ll_queue(self):
        if self.obj.is_empty():
            return f"{self.ds_class}{self.total_nodes}: []"
        return f"{self.ds_class}{self.total_nodes}: {self.front_marker}{self.elements}{self.rear_marker}"

    def repr_ll_queue(self) -> str:
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.front_element}{self.rear_element}"

class CircArrayQueueRepr(BaseRepr):
    """Linked list queue representation"""

    @property
    def storage(self):
        return f"[{self.obj.queue_size}/{self.obj._capacity}]"

    @property
    def front_marker(self):
        color_front = self._ansi.color(f"(F)", Ansi.GREEN)
        return f"{color_front}"

    @property
    def rear_marker(self):
        color_rear = self._ansi.color(f"(R)", Ansi.GREEN)
        return f"{color_rear}"

    @property
    def buffer_type(self):
        if self.obj.overwrite:
            return f"[Buffer=Overwrite]"
        else:
            return f"[Buffer=Static]"

    @property
    def front_element(self):
        front = self.obj.front
        color_front = self._ansi.color(f"{front}", Ansi.GREEN)
        return f"[Front={color_front}]"

    @property
    def rear_element(self):
        rear = self.obj.rear
        color_rear = self._ansi.color(f"{rear}", Ansi.GREEN)
        return f"[Rear={color_rear}]"

    @property
    def elements(self):
        def _element_generator(color=Ansi.GREEN):
            """colors the front and rear in a specified color"""
            for i in range(self.obj.queue_size):
                index = (self.obj._front + i) % self.obj._capacity
                value = self.obj._buffer.array[index]

                if value in (self.obj.front, self.obj.rear):
                    yield self._ansi.color(f"{value}", color)
                else:
                    yield str(value)
        elements = f"[{', '.join(_element_generator())}]"
        return elements

    def str_circ_array_queue(self):
        if self.obj.is_empty():
            return f"{self.ds_class}{self.storage}: []"
        return f"{self.ds_class}{self.storage}: {self.front_marker}{self.elements}{self.rear_marker}"

    def repr_circ_array_queue(self) -> str:
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.front_element}{self.rear_element}{self.buffer_type}"

# deques
class CircDequeRepr(BaseRepr):

    @property
    def storage(self):
        return f"[{self.obj.deque_size}/{self.obj.deque_capacity}]"

    @property
    def front_marker(self):
        color_front = self._ansi.color(f"(F)", Ansi.GREEN)
        return f"{color_front}"

    @property
    def rear_marker(self):
        color_rear = self._ansi.color(f"(R)", Ansi.GREEN)
        return f"{color_rear}"

    @property
    def front_element(self):
        front = self.obj.front
        color_front = self._ansi.color(f"{front}", Ansi.GREEN)
        return f"[Front={color_front}]"

    @property
    def rear_element(self):
        rear = self.obj.rear
        color_rear = self._ansi.color(f"{rear}", Ansi.GREEN)
        return f"[Rear={color_rear}]"

    @property
    def elements(self):
        def _element_generator(color=Ansi.GREEN):
            """colors the front and rear in a specified color"""
            for i in range(self.obj.deque_size):
                index = (self.obj._front + i) % self.obj.deque_capacity
                value = self.obj._buffer.array[index]

                if value in (self.obj.front, self.obj.rear):
                    yield self._ansi.color(f"{value}", color)
                else:
                    yield str(value)

        elements = f"[{', '.join(_element_generator())}]"
        return elements

    def str_circ_deque(self):
        if self.obj.is_empty(): return f"{self.ds_class}{self.storage}: []"
        return f"{self.ds_class}{self.storage}: {self.front_marker}{self.elements}{self.rear_marker}"

    def repr_circ_deque(self) -> str:
        """Displays the memory address and other useful info"""
        if self.obj.is_empty():
            return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}"
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.front_element}{self.rear_element}"

class LlDequeRepr(LinkedListRepr):
    """Linked lIst console visualization"""
    @property
    def front_marker(self):
        color_front = self._ansi.color(f"(F)", Ansi.GREEN)
        return f"{color_front}"

    @property
    def rear_marker(self):
        color_rear = self._ansi.color(f"(R)", Ansi.GREEN)
        return f"{color_rear}"

    @property
    def front_element(self):
        front = self.obj.front
        color_front = self._ansi.color(f"{front}", Ansi.GREEN)
        return f"[Front={color_front}]"

    @property
    def rear_element(self):
        rear = self.obj.rear
        color_rear = self._ansi.color(f"{rear}", Ansi.GREEN)
        return f"[Rear={color_rear}]"

    @property
    def total_nodes(self):
        total_nodes = self.obj.dll.total_nodes
        return f"[{total_nodes}]"

    @property
    def elements(self):
        def _element_generator():
            """colors the front and rear in a specified color"""
            current_node = self.obj._dll.head
            while current_node:
                element = current_node.element
                if element == self.obj.front:
                    yield self._ansi.color(f"{element}", Ansi.GREEN)
                elif element == self.obj.rear:
                    yield self._ansi.color(f"{element}", Ansi.GREEN)
                else:
                    yield str(current_node.element)
                current_node = current_node.next    # traverse

        elements = f"[{', '.join(_element_generator())}]"
        return elements

    def dll_str_deque(self):

        if self.obj.is_empty():
            return f"{self.ds_class}{self.total_nodes}: []"
        return f"{self.ds_class}{self.total_nodes}: {self.front_marker}{self.elements}{self.rear_marker}"

    def dll_repr_deque(self):
        """Displays the memory address and other useful info"""

        if self.obj.is_empty():
            return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}"

        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.front_element}{self.rear_element}"


# priority queues
class PQueueRepr(BaseRepr):

    @property
    def priority_element(self) -> str:
        priority = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)
        return f"[Priority={priority}]"

    @property
    def storage(self) -> str:
        total_elements = self.obj.pqueue_size
        array_capacity = self.obj.data.capacity
        return f"[{total_elements}/{array_capacity}]"

    @property
    def keytype(self) -> str:
        return f"[Keytype={self.obj.keytype.__name__}]"

    @property
    def elements(self) -> str:
        def _generate_items():
            for i in range(self.obj.pqueue_size):
                kv_pair = self.obj._data.array[i]
                priority, element = kv_pair
                # color priority element
                if element == self.obj.priority:
                    yield self._ansi.color(f"[{priority}]: {element}", Ansi.GREEN)
                else:
                    yield f"[{priority}]: {element}"
        elements = f"[{', '.join(_generate_items())}]"
        return elements

    def str_simple_pq(self):
        if self.obj.is_empty():
            return f"{self.ds_class}{self.storage}: []"
        return f"{self.ds_class}{self.storage}: {self.elements}"

    def repr_simple_pq(self):
        """Displays the memory address and other useful info"""
        if self.obj.is_empty():
            return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}"
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.priority_element}{self.keytype}"


# heaps
class BinaryHeapRepr(BaseRepr):

    @property
    def heap_type(self):
        """boolean for min or max heap - info for __str__"""
        if self.obj.heap_type:
            color_heap_type = self._ansi.color(f"Min Heap",Ansi.RED)
            return f"[Heap_Type={color_heap_type}]"
        else:
            color_heap_type = self._ansi.color(f"Max Heap",Ansi.RED)
            return f"[HeapType={color_heap_type}]"

    @property
    def priority_element(self) -> str:
        priority = self._ansi.color(f"{self.obj.priority}", Ansi.GREEN)
        return f"[Priority={priority}]"

    @property
    def storage(self) -> str:
        total_elements = self.obj.pqueue_size
        array_capacity = self.obj.data.capacity
        return f"[{total_elements}/{array_capacity}]"

    @property
    def keytype(self) -> str:
        return f"[Keytype={self.obj.keytype.__name__}]"

    @property
    def elements(self) -> str:
        def _generate_items():
            for i in range(self.obj.pqueue_size):
                kv_pair = self.obj._data.array[i]
                priority, element = kv_pair
                # color priority element
                if element == self.obj.priority:
                    yield self._ansi.color(f"[{priority}]: {element}", Ansi.GREEN)
                else:
                    yield f"[{priority}]: {element}"

        elements = f"[{', '.join(_generate_items())}]"
        return elements

    def str_heap(self):
        # empty case:
        if self.obj.is_empty():
            return f"{self.ds_class}{self.storage}: []"
        return f"{self.ds_class}{self.storage}: {self.elements}"

    def repr_heap(self):
        """Displays the memory address and other useful info"""

        if self.obj.is_empty():
            return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.heap_type}"
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}{self.priority_element}{self.keytype}{self.heap_type}"


# region Maps
class OAHashTableRepr:
    def __init__(self, map_obj) -> None:
        self.obj = map_obj
        self._ansi = Ansi()

    def str_oa_hashtable(self):
        infostring = f"{self.obj.datatype_string}{self.obj.capacity_string}{{{{{self.obj.table_items}}}}}"
        return infostring

    def repr_oa_hashtable(self):
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        stats = f"{class_address}{self.obj.datatype_string}{self.obj.capacity_string}[{self.obj.loadfactor_string}, {self.obj.probes_string}, {self.obj.tombstone_string}, {self.obj.total_collisions_string}, {self.obj.rehashes_string}, {self.obj.avg_probes_string}]"
        return stats

class ChainHashTableRepr:
    def __init__(self, map_obj) -> None:
        self.obj = map_obj
        self._ansi = Ansi()

    def str_chain_hashtable(self):
        items = self.obj.items()
        infostring = f"[{self.obj.datatype.__name__}]{{{{{str(', '.join(f'{k}: {v}' for k, v in items))}}}}}"
        return infostring

    def repr_chain_hashtable(self):
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        capacity = f"{self.obj.total_elements}/{self.obj.table_capacity}"
        return f"{class_address}, Type: {datatype}, Capacity: {capacity}"

# endregion


# Trees
class TreeNodeRepr:
    def __init__(self, tree_node_obj) -> None:
        self.obj = tree_node_obj
        
    def repr_tnode(self):
        """Displays the memory address and other useful info"""
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        node_status = self.obj.alive
        return f"{class_address}, Type: {datatype}, Node Data: {self.obj.element}, Node Alive?: {node_status}"

    def str_tnode(self):
        datatype = self.obj.datatype.__name__
        class_name = self.obj.__class__.__qualname__
        return f"({class_name}: {datatype}) {self.obj.element}"

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
            hierarchy.append(f"{prefix}{indicator}{str(node.element)}") 

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
        return f"{class_address}, Type: {datatype}, Node Data: {self.obj.element}, Children: L: {left_child} R: {right_child} Node Alive?: {node_status}"

class BinaryTreeRepr:
    def __init__(self, tree_obj) -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    def str_binary_tree(self):
        """binary tree __str__"""
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
        """__repr__ for binary tree"""
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        total_elements =  f"Total Nodes: {len(self.obj)}"
        return f"{class_address}, Type: {datatype}, {total_elements}"


class BSTRepr:
    def __init__(self, tree_obj) -> None:
        self.obj = tree_obj
        self._ansi = Ansi()

    def str_bst(self):
        """ __str__ for binary search tree"""
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
                indicator = "" if prefix == "" else ("└─ " if is_last else "├─ ")

            # add to final string output
            node_string = f"K:{node.key}: {node.element}"
            hierarchy.append(f"{prefix}{indicator}{str(node_string)}")

            # Build prefix for children - Vertical bars "│" are inherited from ancestors that are not last children
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Iterates over the node’s children in reverse. (left to right) --- enumerate gives index i for calculating new prefix.
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
        title = self._ansi.color(f"Binary Search Tree: Inorder Traversal:🌲", Ansi.GREEN)

        return f"\n{title}\nTotal Nodes: {total_tree_nodes}, Tree Height: {tree_height}\n{node_structure}\n"

    def repr_bst(self):
        """ __repr__ for binary search tree"""
        class_address = (f"<{self.obj.__class__.__qualname__} object at {hex(id(self.obj))}>")
        datatype = self.obj.datatype.__name__
        total_elements =  f"Total Nodes: {len(self.obj)}"
        return f"{class_address}, Type: {datatype}, {total_elements}"


# Graphs
