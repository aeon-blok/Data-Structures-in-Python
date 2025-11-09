from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator, Generator, Iterable
from abc import ABC, ABCMeta, abstractmethod

# region custom imports
from utils.custom_types import T
from utils.validation_utils import enforce_type
from utils.representations import str_array, repr_array, str_view, repr_view
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

    def __init__(self, element: T) -> None:
        self._element = element
        # initialized as none, stores a reference to the next node in the linked list
        self._next: Optional["Node[T]"] = None  

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

    def __repr__(self) -> str:
        return f"Node: {self._element}"


class LinkedList(LinkedListADT[T], CollectionADT[T]):
    """Implements a Singly Linked List with an additional tail node (for O(1) tail insertions)"""

    def __init__(self) -> None:
        self._head: Optional[Node] = None  # first node in the linked list
        self._tail: Optional[Node] = None
        self._total_nodes: int = 0  # tracks the number of nodes in the linked list

    # ----- Utility Operations -----

    def is_list_empty(self):
        if self.is_empty():
            raise ValueError(f"Error: The Linked list is empty. Total Nodes: {self._total_nodes}")

    def __str__(self) -> str:
        """Displays all the content of the linked list as a string."""

        seperator = " ->> "

        if self._head is None:
            return f"List is Empty"

        def _simple_traversal():
            """traverses the nodes and returns a string via generator"""
            current_node = self._head
            while current_node:
                yield str(current_node._element)
                current_node = current_node.next
                if current_node == self._head:
                    break

        infostring = f"[head]{seperator.join(_simple_traversal())}[tail]"

        return infostring

    # ----- Canonical ADT Operations -----
    # ----- Accessor Operations -----
    def head(self):
        """returns the head node if it exists for use as a reference."""
        self.is_list_empty()
        return self._head

    def tail(self):
        """returns the tail node if it exists for use as a reference"""
        self.is_list_empty()
        return self._tail

    def __iter__(self):
        """Allows iteration over the Nodes in the linked list. (via for loops etc)"""
        current_node = self._head
        while current_node:
            yield current_node._element
            current_node = current_node.next

    # ----- Mutator Operations -----
    def insert_head(self, element):
        """
        This method inserts a new node at the start(head) of the linked list and returns the node as a reference.
        """

        new_node = Node(element)  # create new node object with user data
        new_node._next = self._head  # insert before head
        self._head = new_node  # the new node becomes the new head
        self._total_nodes += 1  # update the linked list elements tracker

        # for empty list - head and tail point to the new node
        if self._tail is None:
            self._tail = new_node

        return new_node # returns the node for reference

    def insert_tail(self, element):
        """Inserts a new node at the end (tail) of the linked list."""
        new_node = Node(element)

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
        """Inserts a new element value & node, after a user specified node reference."""
        new_node = Node(element)
        ref_node = node

        # if ref node is the tail - new node becomes the tail.
        # also works for 1 member llist - head is the tail.
        if node == self._tail:
            node.next = new_node
            self._tail = new_node
        else:
            # otherwise add to chain
            new_node.next = ref_node.next   
            ref_node.next = new_node
        self._total_nodes += 1
        return new_node

    def insert_before(self, node, element):
        """Inserts a new Element(value) and node before a specified node reference."""
        new_node = Node(element)
        ref_node = node

        # if ref node is the head - become new head
        if ref_node == self._head:
            new_node.next = self._head
            self._head = new_node
        else:
            # otherwise traverse linked list and add to chain.
            current_node = self._head
            while current_node:
                if current_node.next == ref_node:
                    current_node.next = new_node
                    new_node.next = ref_node
                    break
                # traverse linked list
                current_node = current_node.next

        self._total_nodes += 1
        return new_node

    def replace(self, node, element):
        """replaces the element(value) at a specific node reference. - returns the old (replaced) value!"""
        old_value = node.element
        node.element = element
        return old_value

    def delete(self, node):
        """removes a specific Node via node reference O(1) - Use a doubly linked list."""
        old_value = node.element
        # if the node is the head - o(1)
        if node == self._head:
            return self.delete_head()
        # deleting tail node... -for singly linked list - o(n)
        elif node == self.tail:
            return self.delete_tail()
        else:
            # delete o(1) in singly linked list via pointer manipulation.
            # essentially we are shifting its neighbours values into the current node. similar to array deletions
            node.element = node.next.element
            # then we skip the next node, simulating a deletion, garbage collection will delete the unreferenced node.
            # update target node next pointer to skip future node, and point to 1 after.
            node.next = node.next.next
            self._total_nodes -= 1
        return old_value

    def delete_head(self) -> T:
        """Deletes the Node at the head - and returns the data for inspection"""
        self.is_list_empty() # existence check
        removed_head = self._head._element
        self._head = self._head.next
        self._total_nodes -= 1
        # Linked List had 1 Member: if list becomes empty after deletion - delete tail also.
        if self._head is None:
            self._tail = None
        # return removed node data
        return removed_head

    def delete_tail(self) -> T:
        """Delete the tail node - in singly linked list this is O(N)"""
        self.is_list_empty() # existence check

        old_value = self._tail.element

        # if list had 1 member. list will become empty.
        if self._head == self._tail:
            self._head = self._tail = None
            self._total_nodes -= 1
            return old_value

        # traverse to 1 before tail.
        current_node = self._head
        while current_node.next != self._tail:
            current_node = current_node.next
        current_node.next = None    # dereference old tail
        # replace tail
        self._tail = current_node
        self._total_nodes -= 1

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
            if current_node._element == element:
                return True
            current_node = current_node.next
        return False







# Main -- Client Facing Code ---
def main():
    # ====== TESTING LINKED LIST ======

    linklist = LinkedList[Any]()


if __name__ == "__main__":
    main()
