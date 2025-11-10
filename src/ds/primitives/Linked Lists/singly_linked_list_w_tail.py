from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator, Generator, Iterable, Type
from abc import ABC, ABCMeta, abstractmethod

# region custom imports
from utils.helpers import RandomClass
from utils.custom_types import T
from utils.validation_utils import enforce_type
from utils.representations import str_ll_node, repr_sll_node, str_ll, repr_ll
from utils.linked_list_utils import validate_node, assert_list_not_empty, find_node_before_reference, assert_reference_node_exists
from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode


# endregion
"""
**Linked List** is a linear data structure where elements (called nodes) are connected using pointers/references rather than stored in contiguous memory like an array.

Properties:
Ordered: maintains linear sequence of elements.
Dynamic size: grows or shrinks at runtime.
Non-contiguous memory: nodes allocated individually.
Sequential access: access by index requires traversal (O(n)).
"""


# Concrete Classes
class Node(iNode[T]):
    """Node Element in Linked List"""

    def __init__(self, element: T, is_linked: bool = False, list_owner=None) -> None:
        self._element = element
        # initialized as none, stores a reference to the next node in the linked list
        self._next: Optional["Node[T]"] = None
        self._is_linked = is_linked # checks if node is deleted or not.
        # ensures the node belongs to the correct list, preventing cross-list misuse.
        self._list_owner = list_owner


    @property
    def element(self):
        return self._element
    @element.setter
    def element(self, value):
        self._element = value
    @property
    def next(self):
        return self._next
    @next.setter
    def next(self, value) -> None:
        self._next = value
    @property
    def is_linked(self):
        return self._is_linked
    @is_linked.setter
    def is_linked(self, value):
        self._is_linked = value
    @property
    def list_owner(self):
        return self._list_owner
    @list_owner.setter
    def list_owner(self, value):
        self._list_owner = value

    # ----- Utility Operations -----
    def __str__(self):
        return str_ll_node(self)

    def __repr__(self):
        return repr_sll_node(self)


class LinkedList(LinkedListADT[T], CollectionADT[T], Generic[T]):
    """Implements a Singly Linked List with an additional tail node (for O(1) tail insertions)"""

    def __init__(self, datatype: Type[T]) -> None:
        self._head: Optional[Node] = None  # first node in the linked list
        self._tail: Optional[Node] = None
        self._total_nodes: int = 0  # tracks the number of nodes in the linked list
        self._datatype = datatype

    @property
    def datatype(self):
        return self._datatype
    @property
    def total_nodes(self):
        return self._total_nodes
    @property
    def head(self):
        """returns the head node if it exists for use as a reference."""
        assert_list_not_empty(self)
        return self._head
    @property
    def tail(self):
        """returns the tail node if it exists for use as a reference"""
        assert_list_not_empty(self)
        return self._tail

    # ----- Utility Operations -----

    def __str__(self) -> str:
        """Displays all the content of the linked list as a string."""
        return str_ll(self, " ->> ")

    def __repr__(self) -> str:
        return repr_ll(self)

    # ----- Canonical ADT Operations -----
    # ----- Accessor Operations -----

    def __iter__(self) -> Generator[T, None, None]:
        """Allows iteration over the Nodes in the linked list. (via for loops etc)"""
        current_node = self._head
        while current_node:
            yield current_node._element
            current_node = current_node.next

    def search_by_index(self, index: int) -> Optional[iNode[T]]:
        """Finds the Node at a specific index"""
        if index < 0 or index >= self._total_nodes:
            raise IndexError("Index Out of Bounds")

        current_node = self._head
        # node at current index position
        for i in range(index):
            current_node = current_node.next
        return current_node

    def search_for_index(self, element: T) -> Optional[int]:
        """Return the index position of the first node that contains the value."""
        # initialize
        current_node = self._head
        index = 0
        while current_node:
            if current_node.element == element:
                return index
            current_node = current_node.next
            index += 1
        return None # if data not found return None

    # ----- Mutator Operations -----
    def insert_head(self, element):
        """
        This method inserts a new node at the start(head) of the linked list and returns the node as a reference.
        Case 1: Empty List (becomes 1 member list, head and tail are the same node)
        Case 2: replace current head with new node. rereference head
        """

        enforce_type(element, self._datatype)
        new_node = Node(element, is_linked=True, list_owner=self)  # create new node object with user data
        new_node.next = self._head  # insert before head
        self._head = new_node  # the new node becomes the new head
        self._total_nodes += 1  # update the linked list elements tracker

        # for empty list - head and tail point to the new node
        if self._tail is None:
            self._tail = new_node

        return new_node # returns the node for reference

    def insert_tail(self, element):
        """
        Inserts a new node at the end (tail) of the linked list.
        Case 1: Empty List (becomes 1 member list, both head and tail are the same node)
        Case 2: replace current tail - rereference tail
        """

        enforce_type(element, self._datatype)
        new_node = Node(element, is_linked=True, list_owner=self)

        if self._tail:
            self._tail.next = new_node  # tail now points to the new last node.
            self._tail = new_node  # update last node with new node

        # if empty list - both head and tail point to the same node
        else:
            self._head = new_node
            self._tail = new_node

        # update the linked list size tracker
        self._total_nodes += 1

        return new_node

    def insert_after(self, node, element):
        """
        Inserts a new element value & node, after a user specified node reference.
        Case 1: Insert After Tail (also works for 1 member list)
        Case 2: Insert After Middle
        Case 3: Node Not Found
        """
        assert_list_not_empty(self)
        validate_node(self, node, iNode)
        enforce_type(element, self._datatype)

        new_node = Node(element, is_linked=True, list_owner=self)
        ref_node = node

        # Insert After Middle Case: otherwise add to chain
        new_node.next = ref_node.next # link to 1 after ref  
        ref_node.next = new_node    # link ref to new node

        # Insert After Tail Case: if ref node is the tail - new node becomes the tail. -- also works for 1 member llist - head is the tail.
        if ref_node == self._tail:
            self._tail = new_node

        self._total_nodes += 1
        return new_node

    def insert_before(self, node, element):
        """
        Inserts a new Element(value) and node before a specified node reference.
        In this specific scenario, traversal is necessary because a singly linked list's nodes only point forward, so you must find the node immediately before the reference node to update its next pointer
        Case 1: Empty List
        Case 2: Insert Before Head
        Case 3: Insert Before Middle (handles tail also)
        Case 4: Node Not Found. (use a boolean flag)
        """

        # Handle Empty List Case:
        assert_list_not_empty(self)

        # validate
        validate_node(self, node, iNode)
        enforce_type(element, self._datatype)

        # initialize nodes
        new_node = Node(element, is_linked=True, list_owner=self)
        ref_node = node

        # Handle Insert Before Head Case: if ref node is the head - become new head -- O(1)
        if ref_node == self._head:
            new_node.next = self._head
            self._head = new_node
        else:
            # Handle Insert Before Middle Case: traverse linked list and add to chain. -- O(n)
            current_node = find_node_before_reference(self, ref_node)
            assert_reference_node_exists(current_node, ref_node)
            # insert new node into chain
            current_node.next = new_node
            new_node.next = ref_node

        self._total_nodes += 1
        return new_node

    def replace(self, node, element):
        """replaces the element(value) at a specific node reference. - returns the old (replaced) value!"""
        assert_list_not_empty(self)
        enforce_type(element, self._datatype)
        validate_node(self, node, iNode)

        old_value = node.element
        node.element = element

        return old_value

    def delete(self, node):
        """removes a specific Node via node reference O(N) - Use a doubly linked list for O(1)."""

        assert_list_not_empty(self)
        validate_node(self, node, iNode)

        old_node = node
        old_value = node.element

        # if the node is the head - o(1)
        if old_node == self._head:
            return self.delete_head()
        # deleting tail node... -for singly linked list - o(n)
        elif old_node == self._tail:
            return self.delete_tail()
        else:
            # traverse to node reference
            current_node = find_node_before_reference(self, old_node)

            # Case: ref node is tail:
            assert_reference_node_exists(current_node, old_node)

            # update pointers to unlink ref node
            current_node.next = node.next
            # update Node tracking parameters (no longer belongs to a list or is linked.)
            old_node.is_linked = False
            old_node.list_owner = None
            old_node.next = None

            self._total_nodes -= 1
            return old_value

    def delete_head(self) -> T:
        """
        Deletes the Node at the head - and returns the data for inspection
        Case 1: Empty List (raise exception)
        Case 2: 1 Member List (Head is Tail) - delete last member and make list empty
        Case 3: replace current head with its neighbour
        """
        assert_list_not_empty(self)
        old_head = self._head
        old_head_element = self._head.element

        self._head = self._head.next
        self._total_nodes -= 1

        # Linked List had 1 Member: if list becomes empty after deletion - delete tail also.
        if self._head is None:
            self._tail = None

        # update Node tracking parameters (no longer belongs to a list or is linked.)
        old_head.is_linked = False
        old_head.list_owner = None
        old_head.next = None

        # return removed node data
        return old_head_element

    def delete_tail(self) -> T:
        """Delete the tail node - in singly linked list this is O(N)"""
        assert_list_not_empty(self)

        old_tail = self._tail
        old_value = self._tail.element

        # if list had 1 member. list will become empty.
        if self._head == self._tail:
            self._head = self._tail = None
        else:
            # traverse to 1 before tail.
            current_node = self._head
            while current_node.next != self._tail:
                current_node = current_node.next

            current_node.next = None    # dereference old tail
            # replace tail
            self._tail = current_node

        self._total_nodes -= 1
        # update Node tracking parameters (no longer belongs to a list or is linked.)
        old_tail.is_linked = False
        old_tail.list_owner = None
        old_tail.next = None

        return old_value

    # ----- Meta Collection ADT Operations -----
    def clear(self):
        while not self.is_empty():
            self.delete_head()

    def is_empty(self) -> bool:
        return self._head is None

    def __len__(self) -> int:
        return self._total_nodes

    def __contains__(self, element) -> bool:
        """return True or False if a node contains the specified data."""
        current_node = self._head
        while current_node:
            if current_node.element == element:
                return True
            current_node = current_node.next
        return False


# Main -- Client Facing Code ---

def main():
    # ====== Input Data ====== 
    # ====== TESTING LINKED LIST ====== 

    sll = LinkedList(str)
    print(repr(sll))

    print(f"\nInsert Node at Head")
    node_a = sll.insert_head("Begninnings")
    print(sll)

    print(f"\nInsert Node at Tail")
    node_b = sll.insert_tail("Endings")
    print(sll)

    print(f"\nInserting after head")
    node_c = sll.insert_after(node_a, "the number 255")
    print(sll)

    print(f"\nInserting after tail")
    node_d = sll.insert_after(node_b, "the new tail")
    print(sll)

    print(f"\nInsert after random middle segment")
    node_e = sll.insert_after(node_c, "i was a stone")
    print(sll)

    print(f"\nInsert before head")
    node_f = sll.insert_before(node_a, "New Head Tuple")
    print(sll)

    print(f"\nInsert before tail")
    node_g = sll.insert_before(node_d, "Falseylicious")
    print(sll)

    print(f"\nInsert before random middle segment")
    node_h = sll.insert_before(node_e, "another randm string")
    print(sll)

    print(f"\nreplacing element(value) of a node.")
    replaced_node_h_value = sll.replace(node_h, "REPLACED")
    print(sll)

    print(f"\nDeleting head node")
    deleted_new_head_value = sll.delete_head()
    print(sll)

    print(f"\nDeleting tail node")
    deleted_tail_value = sll.delete_tail()
    print(sll)

    print(f"Deleting a Middle Node")
    deleted_middle_value = sll.delete(node_c)
    print(sll)

    print(f"\nTesting node validation Step A- invalid node type")
    try:
        testnode = sll.insert_after(replaced_node_h_value, "should not work")
    except Exception as e:
        print(f"{e}")
    print(sll)

    print(f"\nTesting node validation Step B - referencing a deleted node")
    print(f"Deleted Node: {repr(node_f)}")
    print(f"Linked Node: {repr(node_a)}")
    print(f"results:")
    try:
        testnode = sll.insert_after(node_f, "should not work")
    except Exception as e:
        print(f"{e}")

    print(f"\nTesting node validation Step C - referencing a node from another list")
    newlinklist = LinkedList(int)
    rando = newlinklist.insert_head(245555)
    another = newlinklist.insert_head(55)
    one = newlinklist.insert_head(99)
    nodeyguy = newlinklist.insert_head(11112)

    try:
        seplistinsert = sll.insert_after(rando, "matrix!")
    except Exception as e:
        print(f"{e}")

    print(f"\nTesting Type Enforcement.")
    try:
        different_type = sll.insert_head(RandomClass("RANDOM"))
    except Exception as e:
        print(f"{e}")

    print(f"\nTesting __len__.")
    print(f"The List has: {len(sll)} Nodes:")
    print(f"{repr(sll)}")

    print(f"\nTesting __contains__")
    print(sll)
    print(f"{sll.__contains__('REPLACED')}")
    print(f"{sll.__contains__('RANDOM')}")

    print(f"\nAccessing Head")
    current_head = sll.head
    print(f"The current head is: {current_head}")
    print(sll)

    print(f"\nAccessing Tail")
    current_tail = sll.tail
    print(f"The current tail is: {current_tail}")
    print(sll)

    print(f"\nTesting Iteration...")
    for i, item in enumerate(sll):
        print(f"{i}: {item}")
    print(sll)

    print(f"\nTesting is empty?")
    print(f"Is the list empty?: {sll.is_empty()}")

    print(f"\nTesting clear")
    sll.clear()
    print(sll)

    print(f"\nIs the list empty?: {sll.is_empty()}")
    print(sll)

if __name__ == "__main__":
    main()
