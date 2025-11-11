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
# endregion

# region custom imports
from utils.helpers import RandomClass
from utils.custom_types import T
from utils.constants import DLL_SEPERATOR
from utils.validation_utils import enforce_type, index_boundary_check
from utils.representations import str_ll_node, repr_dll_node, repr_sll_node, str_ll, repr_ll
from utils.linked_list_utils import (
    validate_node,
    assert_list_not_empty,
    find_node_before_reference,
    assert_reference_node_exists,
    check_node_after_exists,
    check_node_before_exists
)
from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode
from ds.primitives.Linked_Lists.ll_nodes import Dll_Node

# endregion

"""
A Doubly Linked List (DLL) has both a previous and a next pointer.
Can move forwards or backwards in the list. 
"""


# Double Linked List
class DoublyLinkedList(LinkedListADT[T], Generic[T]):
    def __init__(self, datatype: type) -> None:
        self._head: Optional["iNode[T]"] = None
        self._tail: Optional["iNode[T]"] = None
        self._total_nodes: int = 0
        self._datatype = datatype

    @property
    def head(self):
        assert_list_not_empty(self)
        return self._head
    @property
    def tail(self):
        assert_list_not_empty(self)
        return self._tail
    @property
    def datatype(self):
        return self._datatype
    @property
    def total_nodes(self):
        return self._total_nodes

    # ------------ Utilities ------------
    def __reversed__(self):
        """Python Built in - reverses iteration"""
        current_node = self._tail
        while current_node:
            yield current_node._element
            current_node = current_node.prev

    def __getitem__(self, key: iNode[T] | int) -> T:
        """returns the node in the linked list. Overrides builtin: array style index search"""
        # if the key is a node - just return the element. (o(1))
        if isinstance(key, iNode):
            return key.element
        # otherwise we have to adaptive search (o(n/2))
        elif isinstance(key, int):
            item = self.search_index(key)
            return item.element
        else:
            raise TypeError(f"Error: Invalid Key Provided. Please use a Node or an Index Number.")

    def __setitem__(self, key: iNode[T] | int, value: T) -> None:
        """sets the value of a node in the linked list. Overrides builtin: array style index search O(n)"""
        if isinstance(key, iNode):
            key.element = value
        elif isinstance(key, int):
            item = self.search_index(key)
            item.element = value
        else:
            raise TypeError(f"Error: Invalid Key Provided. Please use a Node or an Index Number.")

    def __str__(self) -> str:
        """Displays all the content of the linked list as a string."""
        return str_ll(self, DLL_SEPERATOR)

    def __repr__(self) -> str:
        """For Devs"""
        return repr_ll(self)

    # ----- Meta Collection ADT Operations -----
    def __iter__(self) -> Generator[T, None, None]:
        """Builtin: List can now be iterated over in loops etc..."""
        current_node = self._head
        while current_node:
            yield current_node._element
            current_node = current_node._next

    def __len__(self) -> int:
        """Override Python Built in to ensure the nodes list is returned"""
        return self._total_nodes

    def __contains__(self, value) -> bool:
        """return True or False if a node contains the specified data."""
        current_node = self._head
        while current_node:
            if current_node._element == value:
                return True
            current_node = current_node._next
        return False

    def clear(self) -> None:
        """Deletes all items from the linked list"""
        while not self.is_empty():
            self.delete_head()

    def is_empty(self) -> bool:
        """Boolean Check if the list is empty"""
        return self._head is None

    # ----- Canonical ADT Operations -----
    # ----- Accessor Operations -----
    def search_value(self, element: T, reverse: bool=False) -> Optional["iNode[T]"]:
        """Finds the first value that matches a node -- O(N)"""
        assert_list_not_empty(self)
        enforce_type(element, self.datatype)

        current_node = self._tail if reverse else self._head
        while current_node:
            if current_node._element == element:
                return current_node 
            current_node = current_node._prev if reverse else current_node._next
        return None

    def bidirectional_search_value(self, element: T) -> Optional["iNode[T]"]:
        """
        Bidirectional Search: Average O(N/2), worst O(N) Return the first or last node containing the value, or None if not found.
        Bidirectional traversal improves latency for early exits, not throughput for full scans.
        """
        # empty list Case:
        if self._head is None:
            return None

        # initialize starter nodes
        left = self._head
        right = self._tail

        # Existence check and crossover check
        while (left and right) and (left != right._next):
            if left._element == element:
                return left
            if right._element == element:
                return right
            # move to next step
            left = left._next
            right = right._prev
        return None  # No value found

    def search_all_values(self, element: T) -> Generator["iNode[T]", None, None]:
        """iterate and yield all nodes that contain a value or None if not found..."""
        current_node = self._head
        while current_node:
            if current_node.element == element:
                yield current_node
            current_node = current_node.next

    def search_for_index_by_value(self, element: T) -> Optional[int]:
        """Return the index of the first node with the value, or None if not found."""
        current_node = self._head
        index = 0
        while current_node:
            if current_node._element == element:
                return index
            index += 1
            current_node = current_node.next
        return None

    def search_index(self, index: int) -> Optional["iNode[T]"]:
        """Average O(N/2) -- Adaptive Index Search: Searches for a specific index in the linked list and returns the node for further manipulation"""

        index_boundary_check(index, self._total_nodes)
        assert_list_not_empty(self)

        # if the index is less than half of the list size - start from the head
        if index < self._total_nodes // 2:
            current_node = self._head
            for _ in range(index):
                current_node = current_node.next

        # otherwise start from the tail:
        else:
            current_node = self._tail
            for _ in range(self._total_nodes - 1, index, -1):
                current_node = current_node.prev

        return current_node

    # ----- Mutator Operations -----
    def insert_head(self, element):
        """add a new node at the very beginning of the list — making it the new head."""
        enforce_type(element, self.datatype)

        new_node = Dll_Node(element, is_linked =True, list_owner=self)
        new_node.prev = None
        # point to old head (if list is empty = None)
        new_node.next = self._head

        if self._head:
            # update old head prev pointer: (points to the current head - new node)
            self._head.prev = new_node  
        else:
            # Empty list Case: (head and tail are the same Node)
            self._tail = new_node
        # Assign New node to the head
        self._head = new_node

        self._total_nodes += 1 # increment size tracker

        return new_node

    def insert_tail(self, element):
        """insert a node at the end of the list - the tail."""

        enforce_type(element, self.datatype)
        new_node = Dll_Node(element, is_linked=True, list_owner=self)

        # new tail prev should point to old tail.
        new_node.prev = self._tail
        new_node.next = None

        # Tail Exists Case: is there an exsiting tail. (if so should point to new tail)
        if self._tail:
            self._tail._next = new_node
        # Empty List Case: if list is empty - head and tail are same so insert at head and tail
        else:
            self._head = new_node

        # insert new node at tail
        self._tail = new_node

        # increment size tracker
        self._total_nodes += 1

        return new_node

    def insert_after(self, node, element):
        """Inserts a new node after a specific node -- O(1)"""

        assert_list_not_empty(self)
        validate_node(self, node, iNode)
        enforce_type(element, self.datatype)

        new_node = Dll_Node(element, is_linked=True, list_owner=self)
        # Step 1: link the new node to the previous node
        new_node.prev = node
        # Step 2: link the new node to the future node
        new_node.next = node.next
        # Step 3: link the future node to the new node
        if node.next:
            node.next.prev = new_node
        else:
            # if the future node doesnt exist. - assign to tail(last node)
            self._tail = new_node
        # Step 4: link the previous node to the new node
        node.next = new_node

        self._total_nodes += 1  # increment size tracker
        return new_node

    def insert_before(self, node, element):
        """Inserts a new node before a specific node -- O(1)"""

        assert_list_not_empty(self)
        validate_node(self, node, iNode)
        enforce_type(element, self.datatype)
        new_node = Dll_Node(element, is_linked=True, list_owner=self)

        # Step 1: link the new node to the future node
        new_node._next = node
        # Step 2: link the new node to the previous node
        new_node._prev = node.prev
        # Step 3: link the previous node to the new node
        if node.prev:
            node.prev.next = new_node
        # if there is no previous node - assign to the head(start)
        else:
            self._head = new_node

        # Step 4: link the future node to the new node
        node.prev = new_node

        self._total_nodes += 1  # increment size tracker
        return new_node

    def replace(self, node, element):
        """replace value of a specific node. returns the old value."""
        assert_list_not_empty(self)
        validate_node(self,node,iNode)
        enforce_type(element, self.datatype)
        old_value = node.element
        node.element = element
        return old_value

    def delete(self, node):
        """Delete a node via reference -- O(1)"""
        # Empty List Case:
        assert_list_not_empty(self)
        validate_node(self, node, iNode)

        # initialize nodes
        old_node = node
        old_value = node.element
        previous_node = old_node.prev
        future_node = old_node.next

        # 1 member list Case: (head is tail)
        if old_node == self._head and old_node == self.tail:
            return self.delete_head()
        elif old_node == self._head:
            # is head Case: the ref node is the head.
            return self.delete_head()
        elif old_node == self._tail:
            # is tail Case: (the ref node is the tail)
            return self.delete_tail() 
        else:
            # middle node Case: (ref node is in the middle of the list)
            previous_node.next = future_node
            future_node.prev = previous_node
            old_node.next = None
            old_node.prev = None
            old_node.is_linked = False
            old_node.list_owner = None

        self._total_nodes -= 1

        return old_value

        pass

    def delete_head(self):
        """Deletes Node at the Head Position"""

        assert_list_not_empty(self)
        old_head = self._head
        old_value = self._head.element

        # 1 Member List Case: if there is only 1 node:
        if self._head == self._tail:
            self._head.is_linked = False
            self._head.list_owner = None
            self._head.prev, self._head.next = None, None
            self._head = self._tail = None
        else:
            # Main Case: - move head forward 1 node and delete its previous link
            self._head = old_head.next
            self._head.prev = None
            # dereferencing old head
            old_head.is_linked = False
            old_head.list_owner = None
            old_head.prev, old_head.next = None, None

        self._total_nodes -= 1  # update size tracker
        return old_value

    def delete_tail(self):
        """Deletes the Node at the Tail Position"""

        assert_list_not_empty(self)
        old_tail = self._tail    # for dereferencing
        old_value = self._tail.element

        # 1 member list Case: if there is only 1 node (head & tail)
        if self._head == self._tail:
            self._head = self._tail = None
        else:
            # Tail exists Case: dereference and set future node to none
            self._tail = old_tail.prev
            self._tail.next = None   # tail future node must always be None

        # fully detach the old tail from the list
        old_tail.prev, old_tail.next = None, None
        old_tail.is_linked = False
        old_tail.list_owner = None

        self._total_nodes -= 1  # decrement size counter
        return old_value

    def delete_after(self, node):
        """Deletes Node that comes after a specified Node"""

        assert_list_not_empty(self)
        validate_node(self, node, iNode)
        check_node_after_exists(node)

        # Step 1: Set Target for deletion
        old_node = node.next
        old_value = node.next.element
        future_node = old_node.next

        # Step 2: Update Node to point to future node (the node after the deleted node)
        node.next = future_node
        # Step 3: Middle Node Case: if there is a future node (after deleted node): link back to ref node
        if future_node:
            future_node.prev = node
        # Step 3B: is tail Case: update Node to become the tail
        else:
            self._tail = node
        # Step 4: dereference deleted node
        old_node.next, old_node.prev = None, None
        old_node.is_linked = False
        old_node.list_owner = None

        self._total_nodes -= 1  # decrement size tracker
        return old_value

    def delete_before(self, node):
        """Deletes Node that comes before a specified Node"""

        assert_list_not_empty(self)
        validate_node(self,node,iNode)
        check_node_before_exists(node)

        # Step 1: initialize nodes
        old_node = node.prev
        old_value = old_node.element
        previous_node = old_node.prev

        # Step 2: if there is a node before the target, assign its future node to ref node, and reassign ref node's previous node to the 1 before the target
        if previous_node:
            previous_node.next = node
            node.prev = previous_node
        # is head Case: otherwise target node is the head - change to node
        else:
            self._head = node
            node.prev = None    # dereferences any prior links

        # Dereference Deleted Node
        old_node.is_linked = False
        old_node.list_owner = None
        old_node.prev, old_node.next = None, None

        self._total_nodes -= 1
        return old_value


# Main --- Client Facing Code ---

def main():
    # -------------------------
    # Test DoublyLinkedList
    # -------------------------
    dll = DoublyLinkedList(str)
    print(dll)
    print(repr(dll))
    print(f"\nTesting is_empty? {dll.is_empty()}")

    print(f"\nTestng Insertions:")
    print(f"Testing Insert Head")
    head = dll.insert_head("0")
    print(dll)
    node_a = dll.insert_head("1")
    print(dll)
    node_b = dll.insert_tail("2")
    print(dll)
    node_c = dll.insert_after(node_b, "3")
    print(dll)
    node_d = dll.insert_after(head, "4")    
    print(dll)
    node_e = dll.insert_after(node_b, "5")
    print(dll)
    node_f = dll.insert_before(node_c, "6")
    print(dll)
    node_g = dll.insert_before(node_a, "7")
    print(dll)
    node_h = dll.insert_before(node_d, "8")    
    print(dll)
    node_ab = dll.insert_tail("9")
    node_ac = dll.insert_tail("10")
    node_ad = dll.insert_after(node_ab, "11")
    node_ae = dll.insert_head("12")
    node_af = dll.insert_before(node_ac, "13")
    print(repr(node_ae))
    print(dll)
    node_h_delete_value = dll.delete(node_h)
    print(f"Deleted Node: {node_h_delete_value}")
    print(dll)
    node_a_delete_value = dll.delete(node_a)
    print(f"Deleted Node: {node_a_delete_value}")
    print(dll)
    node_c_delete_value = dll.delete(node_c)
    print(f"Deleted Node: {node_c_delete_value}")
    print(dll)
    head_value_delete = dll.delete_head()
    print(f"Deleted Head: {head_value_delete}")
    print(dll)
    tail_value_delete = dll.delete_tail()
    print(f"Deleted Tail: {tail_value_delete}")
    print(dll)
    node_e_delete_value = dll.delete_after(node_b)
    print(f"Deleted Node: {node_e_delete_value}")
    print(dll)
    node_b_delete_value = dll.delete_after(node_d)
    print(f"Deleted Node: {node_b_delete_value}")
    print(dll)
    node_ad_delete_value = dll.delete_before(node_af)
    print(f"Deleted Node: {node_ad_delete_value}")
    print(dll)
    node_d_delete_value = dll.delete_before(node_f)
    print(f"Deleted Node: {node_d_delete_value}")
    print(dll)
    node_ab_delete_value = dll.delete_before(node_af)
    print(f"Deleted Node: {node_ab_delete_value}")
    print(dll)
    for i, item in enumerate(dll):
        print(f"Iterated over: {i}: {item}")
    node_ba = dll.insert_tail("10")
    node_bb = dll.insert_tail("10")
    node_bc = dll.insert_tail("10")
    a, b, c = list(dll.search_all_values("10"))
    print(repr(a))
    print(b)
    print(c)
    print(dll)
    search_a = dll.search_value("10")
    print(repr(search_a))
    search_c = dll.search_value("10", reverse=True)
    print(repr(search_c))

    index_search_0 = dll.search_for_index_by_value("0")
    print(f"Search for value: '0' return an index number: {index_search_0}")
    print(dll)

    retrieve_index_2 = dll.search_index(2)
    print(repr(retrieve_index_2))
    print(dll)

    get_node_at_value_10 = dll.bidirectional_search_value("10")
    print(repr(get_node_at_value_10))
    print(dll)


    print(f"List is: {len(dll)} Nodes in Size.")
    print(f"Is 10 in Linked list? {'10' in dll}")
    print(f"Is 200 in Linked list? {'200' in dll}")

    print(f"Get item at index 2: {dll[1]}")
    print(dll)
    dll[1] = "25"
    print(f"Set item at index 2: {dll[1]}")
    print(dll)

    print(f"Get item via Node reference: {dll[node_g]}")
    dll[node_g] = "1250"
    print(f"Set item via Node reference: {dll[node_g]}")
    print(dll)

    print(f"\nTesting is_empty? {dll.is_empty()}")
    dll.clear()
    print(dll)
    print(f"Testing is_empty? {dll.is_empty()}")


if __name__ == "__main__":
    main()
