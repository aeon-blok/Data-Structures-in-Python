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

class SortedArrayRepr(ArrayRepr):
    """For Sorted Arrays - special type of array. We need to add additional property to not show the key objects."""

    @property
    def items(self) -> str:
        return f"[{', '.join(str(self.obj.array[i].value) for i in range(self.obj.size))}]"

    @property
    def array_type(self) -> str:
        return f"[{'Static' if self.obj.is_static == True else 'Dynamic'}]"


    def repr_array(self):
        """array __repr__ - for devs"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.storage}"

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

# region Stacks
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
# endregion

# region queues
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
# endregion

# region deques
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
# endregion

# region priority queues
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
# endregion

# region Maps
class OAHashTableRepr(BaseRepr):

    @property
    def datatype(self):
        return f"[{self.obj.datatype_string}]"

    def str_oa_hashtable(self):
        return f"{self.ds_class}{self.obj.capacity_string}: {self.obj.table_items}"

    def repr_oa_hashtable(self):
        stats = f"{self.ds_memory_address}{self.datatype}{self.obj.capacity_string}[{self.obj.loadfactor_string}, {self.obj.probes_string}, {self.obj.tombstone_string}, {self.obj.total_collisions_string}, {self.obj.rehashes_string}, {self.obj.avg_probes_string}]"
        return stats

class ChainHashTableRepr(BaseRepr):

    # todo refactor - add OA - type strings (for icons for repr.)

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

# region sets

class HashSetRepr(BaseRepr):

    @property
    def total_elements(self) -> str:
        return f"[{len(self.obj)}]"

    @property
    def elements(self) -> str:
        return f"{{{f', '.join(str(i) for i in self.obj.members)}}}"

    def str_hashset(self):
        return f"{self.ds_class}{self.elements}"

    def repr_hashset(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_elements}"


# endregion

# region Trees
# Trees
class TreeNodeRepr(BaseRepr):

    @property
    def node_status(self):
        if self.obj.alive:
            status = self._ansi.color(f"alive", Ansi.GREEN)
        else:
            status = self._ansi.color(f"deleted", Ansi.RED)
        return f"[status={status}]"

    @property
    def owner(self):
        instance = self.obj.tree_owner
        owner_class = instance.__class__.__name__
        memory_address = hex(id(instance))
        if instance is not None:
            string = f"[owner={owner_class}: {memory_address}]"
        else:
            string = f"[owner=None]"
        return string

    @property
    def parent(self):
        parent = self.obj.parent
        if parent is not None:
            color_parent = self._ansi.color(f"{parent.element}", Ansi.GREEN)
        else:
            color_parent = self._ansi.color(f"None", Ansi.GREEN)
        return f"[parent={color_parent}]"

    @property
    def element(self):
        color_element = self._ansi.color(f"{self.obj.element}", Ansi.BLUE)
        return f"{color_element}"

    @property
    def children(self):
        children = self.obj.num_children()
        return f"[children={children}]"

    def repr_tnode(self):
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.node_status}{self.owner}{self.children}"

    def str_tnode(self):
        return f"{self.element}"

class GenTreeRepr(BaseRepr):

    @property
    def tree_height(self):
        if self.obj.root is None:
            return f"[height=0]"
        else:
            height = self.obj.height(self.obj.root)
        return f"[height={height}]"

    @property
    def tree_depth(self):
        if self.obj.root is None:
            return f"[depth=0]"
        else:
            depth = self.obj.depth(self.obj.root)
        return f"[depth={depth}]"

    @property
    def traversal_type(self):
        traversal = self.obj.traversal
        return f"[traversal={traversal}]"

    @property
    def total_nodes(self):
        number = len(self.obj)
        return f"[total_nodes={number}]"

    def repr_gen_tree(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_height}{self.tree_depth}{self.traversal_type}"
        
    def str_gen_tree(self):
        """
        Traverses the Tree via stack
        adds connector symbols in front of each node value, depending on whether it is the last child "└─" or one of many "├─",
        every node adds either " " if parent is last child (no vertical bar needed) or "| " if parent is not last child (vertical bar continues)
        the node & its display symbols are appended to a list for the final string output.
        """

        # todo add BFS as a flag option here. choose between dfs and bfs representation.

        total_tree_nodes = len(self.obj)
        tree_height = self.obj.height(self.obj.root)

        if self.obj.root is None:
            return f"[🌳 empty tree]"

        hierarchy = []
        tree = [(self.obj.root, "", True)]  # (node, prefix, is_last)
        # tree visualization construction loop (change to stack soon)
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
        stats = f"{self.total_nodes}{self.tree_height}"
        return f"\n{title}\n{stats}\n{node_structure}\n"

class BTreeNodeRepr(BaseRepr):
    """Node representation for Btree Nodes"""

    @property
    def is_leaf(self) -> str:
        return f"[is_leaf={self.obj.is_leaf}]"

    @property
    def capacity(self) -> str:
        return f"[{self.obj.num_keys}/{self.obj.max_keys}]"

    @property
    def items(self) -> str:
        combo = str(', '.join(f'{k}={v}' for k,v in zip(self.obj.keys, self.obj.elements)))
        return f"[{combo}]"

    @property
    def keys_range(self) -> str:
        array_length = len(self.obj.keys)
        begin = self.obj.keys[0] if not self.obj.keys.is_empty() else None
        end = self.obj.keys[array_length-1] if not self.obj.keys.is_empty() else None
        return f"[key range: {begin}|{end}]"

    def str_btree_node(self):
        return f"{self.ds_class}{self.capacity}{self.keys_range}"

    def repr_btree_node(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.capacity}"

class PageRepr(BaseRepr):
    """representation for Page object (in memory representation of disk stored bytes.)"""

    @property
    def page_id(self) -> str:
        return f"[id={self.obj.page_id}]"
    
    @property
    def space(self) -> str:
        return f"[avl_space={self.obj.available_space}]"
    
    def str_page(self):
        return f"{self.ds_class}{self.space}"
    
    def repr_page(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.space}"

class BTreeRepr(BaseRepr):
    """B Tree representation"""

    @property
    def tree_size(self) -> str:
        return f"[keys={self.obj.total_keys}]"

    @property
    def total_nodes(self) -> str:
        return f"[nodes={self.obj.total_nodes}]"

    @property
    def node_capacity(self) -> str:
        return f"[kpn: min={self.obj.min_keys}, max={self.obj.max_keys}]"

    @property
    def tree_height(self) -> str:
        return f"[height={self.obj.tree_height}]"

    @property
    def valid_tree(self) -> str:
        self.obj.validate_tree
        return f"[valid=True]"

    def str_btree(self):
        """DFS preorder visualization of the B-tree showing key ranges per node."""

        if self.obj.root is None:
            return f"[empty tree]"

        hierarchy = self.obj.bfs_view

        node_structure = "\n".join(hierarchy)
        title = f"🌿 B-Tree:"
        stats = f"{self.node_capacity}{self.tree_size}{self.total_nodes}{self.tree_height}{self.valid_tree}"
        return f"\n{title}\n{stats}\n{node_structure}"

    def repr_btree(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_size}{self.node_capacity}"


# region Binary Trees
# Binary Trees
class BinaryNodeRepr(TreeNodeRepr):

    @property
    def children(self):
        if self.obj.left is not None:
            left = self._ansi.color(f"{self.obj.left.element}", Ansi.GREEN)
        else:
            left = self._ansi.color(f"None", Ansi.GREEN)
        if self.obj.right is not None:
            right = self._ansi.color(f"{self.obj.right.element}", Ansi.RED)
        else:
            right = self._ansi.color(f"None", Ansi.RED)
        return f"[children: L={left}, R={right}]"

    @property
    def sibling(self):
        if self.obj.sibling is not None:
            sib = self.obj.sibling.element
        else:
            sib = "None"
        return f"[sibling={sib}]"

    def str_binary_node(self):
        return f"{self.obj.element}"

    def repr_binary_node(self):
        """Displays the memory address and other useful info"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.sibling}{self.children}{self.node_status}{self.owner}"

class BinaryTreeRepr(BaseRepr):

    @property
    def tree_height(self):
        if self.obj.root is None:
            return f"[height=0]"
        else:
            height = self.obj.height()
        return f"[height={height}]"

    @property
    def tree_depth(self):
        if self.obj.root is None:
            return f"[depth=0]"
        else:
            depth = self.obj.depth(self.obj.root)
        return f"[depth={depth}]"

    @property
    def total_nodes(self):
        number = len(self.obj)
        return f"[total_nodes={number}]"

    def str_binary_tree(self):
        """binary tree __str__"""
        total_tree_nodes = len(self.obj)
        tree_height = self.obj.height()

        if self.obj.root is None:
            return f"[🌳 empty tree]"

        hierarchy = []
        tree = [(self.obj.root, "", True)]  # (node, prefix, is_last)

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print nothing
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
        stats = f"{self.total_nodes}{self.tree_height}"
        return f"\n{title}\n{stats}\n{node_structure}\n"

    def repr_binary_tree(self):
        """__repr__ for binary tree"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_height}{self.tree_depth}"
# endregion

# region BST
class BSTNodeRepr(TreeNodeRepr):

    @property
    def children(self):
        if self.obj.left is not None:
            left = self._ansi.color(f"{self.obj.left.element}", Ansi.GREEN)
        else:
            left = self._ansi.color(f"None", Ansi.GREEN)
        if self.obj.right is not None:
            right = self._ansi.color(f"{self.obj.right.element}", Ansi.RED)
        else:
            right = self._ansi.color(f"None", Ansi.RED)
        return f"[children: L={left}, R={right}]"

    @property
    def element(self):
        elem = self.obj.element
        key_value = f"{self.obj.key.value}"
        return f"{elem} [k:{key_value}]"

    def str_bst_node(self):
        return f"{self.element}"

    def repr_bst_node(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.children}{self.node_status}{self.owner}"

class BSTRepr(BaseRepr):

    @property
    def tree_height(self):
        if self.obj.root is None:
            return f"[height=0]"
        else:
            height = self.obj.height()
        return f"[height={height}]"

    @property
    def total_nodes(self):
        number = len(self.obj)
        return f"[total_nodes={number}]"

    def str_bst(self):
        """ __str__ for binary search tree - slight modifications to the code used for other trees."""
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
                # ! this is the code that is modified for BST
                indicator = "└─ " if not (node.left or node.right) else ("└─ " if is_last else "├─ ")

            # add to final string output
            node_string = f"{node.key}: {node.element}"
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
        stats = f"{self.total_nodes}{self.tree_height}"
        return f"\n{title}\n{stats}\n{node_structure}\n"

    def repr_bst(self):
        """ __repr__ for binary search tree"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_height}"


class AVLNodeRepr(BSTNodeRepr):

    @property
    def balance(self) -> str:
        bal = self.obj.balance_factor
        return f"[balance_factor={bal}]"

    @property
    def node_height(self) -> str:
        height = self.obj.height
        return f"[height={height}]"

    @property
    def check_balance(self) -> str:
        return f"[is_balanced?={self.obj.unbalanced}]"

    def str_avl_node(self):
        return f"{self.element}"

    def repr_avl_node(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.check_balance}{self.balance}{self.node_height}{self.children}{self.node_status}{self.owner}"

class AVLTreeRepr(BSTRepr):
    def __init__(self, obj) -> None:
        super().__init__(obj)

    @property
    def unbalanced(self):
        return f"[unbalanced?={self.obj.unbalanced_tree}]"
    
    @property
    def max_bf(self):
        return f"[max_bf={self.obj.max_balance_factor}]"


    def str_avl(self):
        """ __str__ for binary search tree - slight modifications to the code used for other trees."""
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
                # ! this is the code that is modified for BST
                indicator = "└─ " if not (node.left or node.right) else ("└─ " if is_last else "├─ ")

            # add to final string output
            node_string = f"{node.key}: {node.element}"
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
        title = self._ansi.color(f"AVL Tree: 🌲", Ansi.GREEN)
        stats = f"{self.total_nodes}{self.tree_height}{self.unbalanced}{self.max_bf}"
        return f"\n{title}\n{stats}\n{node_structure}\n"

    def repr_avl(self):
        """ __repr__ for AVL Tree"""
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_height}"


class RedBlackNodeRepr(BSTNodeRepr):

    @property
    def black_height(self) -> str:
        return f"[black_height={self.obj.black_height}]"
    
    @property
    def node_color(self) -> str:
        return f"[color={self.obj.color}]"
    
    @property
    def is_leaf(self) -> str:
        return f"[is_leaf?={self.obj.is_leaf()}]"
    
    @property
    def uncle(self) -> str:
        return f"[uncle={self.obj.uncle.element}]" if self.obj.grandparent is not None else "[uncle=None]"

    def str_redblack_node(self):
        return f"{self.ds_class}{self.node_color}{self.element}"
    
    def repr_redblack_node(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.black_height}{self.node_color}{self.is_leaf}{self.uncle}"

class RedBlackTreeRepr(BSTRepr):
    def __init__(self, obj) -> None:
        super().__init__(obj)

    @property
    def black_property(self):
        result = self.obj.is_black_property
        return f"[black_property={result}]"
    
    @property
    def red_property(self):
        result = self.obj.is_red_property
        return f"[red_property={result}]"

    def str_redblack_tree(self):
        """ __str__ for binary search tree - slight modifications to the code used for other trees."""
        total_tree_nodes = len(self.obj)
        tree_height = self.obj.height()
        if self.obj.root == self.obj.NIL:
            return f"[🌳 empty Red Black Tree]"

        hierarchy = []
        tree = [(self.obj.root, "", True)]  # (node, prefix, is_last)

        while tree:
            # we traverse depth-first, which naturally fits a hierarchical print.
            node, prefix, is_last = tree.pop()

            # root (depth = 0), we print 🌲
            if node is self.obj.root:
                indicator = ""

            # ! skip sentinels (in red black tree)
            if node == self.obj.NIL:
                continue

            # decides what connector symbol appears before the node value when printing the tree.
            else:
                # ! this is the code that is modified for BST
                indicator = "└─ " if not (node.left or node.right) else ("└─ " if is_last else "├─ ")

            # add to final string output
            node_string = f"{node.key}: {node.element} ({f'r' if node.is_red else 'b'})"
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
        title = self._ansi.color(f"Red Black Tree: ", Ansi.RED)
        stats = f"{self.total_nodes}{self.tree_height}{self.black_property}{self.red_property}"
        return f"\n{title}\n{stats}\n{node_structure}\n"
    
    def repr_redblack_tree(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_nodes}{self.tree_height}{self.black_property}{self.red_property}"

# endregion

# region Disjoint Set
class AncestorNodeRepr(BaseRepr):
    """representation for Parent Pointer Tree node. also known as ancestor node"""

    @property
    def rank(self) -> str:
        rank = self.obj.rank
        return f"[rank={rank}]"

    @property
    def parent(self) -> str:
        parent = self.obj.parent.element
        return f"[parent={parent}]"

    @property
    def element(self) -> str:
        element = self.obj.element
        return f"[element={element}]"
    
    def str_ancestor_node(self):
        return f"{self.obj.element}"

    def repr_ancestor_node(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.element}{self.rank}{self.parent}"

class DisjointSetForestRepr(BaseRepr):
    """representation of Disjoint set Forest Datat Structure"""

    @property
    def total_sets(self) -> str:
        sets = self.obj.set_count()
        return f"[total sets={sets}]"

    @property
    def reps(self) -> str:
        return f"[representatives={', '.join(f'{i.element}[r={i.rank}]' for i in self.obj.representatives)}]"

    def str_disjoint_set_forest(self):
        return f"{self.ds_class}{self.reps}"

    def repr_disjoint_set_forest(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.total_sets}"
# endregion

# region Graphs
# Graphs

class VertexRepr(BaseRepr):
    """representation for Vertex Nodes"""

    @property
    def element(self) -> str:
        value = self.obj.element
        return f"{value}"

    @property
    def vert_id(self) -> str:
        """uses insert order as an id for the vert."""
        label = self.obj.name
        insert_number = self.obj.insert_order

        # insertion number logic
        if insert_number is None:
            insert_number = f"[_]"
        else:
            insert_number = f"[{insert_number}]"

        # label replaces insertion number
        if label is not None:
            return f"{insert_number}id={label}"
        else:
            return f"{insert_number}"

    def str_vertex(self):
        return f"{self.element}"

    def repr_vertex(self):
        return f"{self.ds_class}{self.element}"

class EdgeRepr(BaseRepr):
    """Edge Object representation"""

    @property
    def weight(self) -> str:
        weight = self.obj.element
        return f"{weight}"

    @property
    def edge_id(self) -> str:
        origin = self.obj.origin.element
        destination = self.obj.destination.element
        return f"{origin} <{self.obj.element}> {destination}"

    def repr_edge(self):
        return f"{self.ds_class}{self.edge_id}"

    def str_edge(self):
        return f"{self.edge_id}"

class GraphRepr(BaseRepr):
    """representation for Graphs"""

    @property
    def directed(self) -> str:
        result = self.obj.is_directed
        if result:
            graph_type = f"directed"
        else:
            graph_type = f"undirected"

        return f"[Mode={graph_type}]"

    @property
    def vertex_count(self) -> str:
        count = self.obj.vertex_count
        return f"[V={count}]"

    @property
    def edge_count(self) -> str:
        count = self.obj.edge_count
        return f"[E={count}]"

    @property
    def adj_map(self) -> str:
        adjacency_map = self.obj.view_adjacency_map
        if self.obj.vertex_count == 0:
            return f"Graph Adjacency Map: Empty Graph..."
        else:
            return self.obj.view_adjacency_map

    def str_graph(self):
        return f"{self.adj_map}"

    def repr_graph(self):
        return f"{self.ds_memory_address}{self.ds_datatype}{self.directed}{self.vertex_count}{self.edge_count}"

# endregion
