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
)
from abc import ABC, ABCMeta, abstractmethod

# endregion

# region custom imports
from utils.helpers import RandomClass
from utils.custom_types import T
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.representations import DllNodeRepr, LinkedListRepr
from adts.collection_adt import CollectionADT
from adts.linked_list_adt import LinkedListADT, iNode

from ds.primitives.Linked_Lists.ll_nodes import Dll_Node
from ds.primitives.Linked_Lists.linked_list_utils import LinkedListUtils


# endregion
"""
Doubly Circular Linked List - where the tail reconnects to the head.

Whats the logic of a Doubly Circular Linked List?
Every Node in the list points in 2 ways - forwards (->>) and backwards (<<-)
Additionally: The Tail points forwards to the Head, connecting the entire list in a loop.

"""

# TODO: Add delete by value
# TODO: add Slice (override __gettitem__)
# TODO: Merging Linked lists (maintaining order)
# TODO: Splitting linked lists
# TODO: add Search all indices (returns multiple indices...)
# TODO: Conditional Traverse
# TODO: Count how many times a value appears in the linked list...
# TODO: remove every N node
# TODO: implement __next__ for generators
# todo Solve simple problems: reverse a linked list, detect a cycle, merge two sorted lists

# TODO: Interview additions:
# TODO: Find middle Node
# TODO: remove duplicates (unsorted - use set), sorted (skip duplicates)
# TODO: Find Nth Node from the end
# TODO: reverse between n and m
# TODO: Partition list
# TODO: swap first and last
# TODO: Swap nodes in pairs....
# TODO: full list reversal - and partial list reversal
# TODO: Palindrome check


class DoublyCircularList(LinkedListADT[T], CollectionADT[T], Generic[T]):
    """Implementation of DCL: """
    def __init__(self, datatype: type) -> None:
        self._head: Optional[iNode[T]] = None
        self._tail: Optional[iNode[T]] = None
        self._total_nodes: int = 0
        self._datatype = datatype
        # composed objects
        self._validators = DsValidation()
        self._utils = LinkedListUtils(self)
        self._desc = LinkedListRepr(self)

    @property
    def head(self):
        return self._head
    @property
    def tail(self):
        return self._tail
    @property
    def datatype(self):
        return self._datatype
    @property
    def total_nodes(self):
        return self._total_nodes

    # ----- Meta Collection ADT Operations -----
    def __iter__(self) -> Generator[T, None, None]:
        """iterates over the linked list and returns element values."""
        for node in self._utils.traverse_dcll_nodes():
            yield node.element

    def __contains__(self, value: T):
        """Does the Linked list contain this value? utilizes bidirectional search O(N/2)"""
        left = self._head
        right = self._tail

        # loop over list from either side
        for _ in range((self._total_nodes + 1) // 2): # half traversal only
            if left.element == value or right.element == value:
                return True

            left = left.next
            right = right.prev

        return False

    def __len__(self):
        return self._total_nodes

    def clear(self):
        """removes all items from the list."""
        while self._total_nodes > 0:
            self.delete_head()

    def is_empty(self) -> bool:
        return self._head is None

    # ------------ Utility ------------
    def __getitem__(self, key: iNode[T] | int) -> T:
        """returns the value of a node in the linked list. Overrides builtin -- array like indexing with a linked list but its O(N)"""
        # if the key is a node - just return the element. (o(1))
        if isinstance(key, iNode):
            return key.element
        # otherwise we have to adaptive search (o(n/2))
        elif isinstance(key, int):
            item = self._search_index(key)
            return item.element
        else:
            raise KeyInvalidError()

    def __setitem__(self, key: iNode[T] | int, value: T) -> None:
        """Overrides python built in - access linked list like an array -- O(N)"""
        """sets the value of a node in the linked list. Overrides builtin: array style index search O(n)"""
        if isinstance(key, iNode):
            key.element = value
        elif isinstance(key, int):
            item = self._search_index(key)
            item.element = value
        else:
            raise KeyInvalidError()

    def __reversed__(self):
        """Python Built in - reverses iteration"""

        current_node = self._tail  

        # empty list Case:
        if not current_node:
            return

        while current_node:
            yield current_node.element
            current_node = current_node.prev
            if current_node == self._tail:
                break

    def __str__(self) -> str:
        """Displays all the content of the linked list as a string."""
        return self._desc.str_ll()

    def __repr__(self) -> str:
        return self._desc.repr_ll()

    # ----- Accessor Operations -----
    def search_value(self, element: T, reverse=False) -> Optional["iNode[T]"]:
        """Finds the first node that matches a value -- O(N)"""
        self._utils.assert_list_not_empty()
        self._validators.enforce_type(element, self.datatype)

        # can go from the head or the tail
        current_node = self._tail if reverse else self._head
        while current_node:
            if current_node.element == element:
                return current_node 
            # traverse the linked list
            current_node = current_node.prev if reverse else current_node.next
            # exit condition
            if current_node == (self._tail if reverse else self._head): 
                break
        return None

    def bidirectional_search_value(self, element) -> Optional["iNode[T]"]:
        """Searches for the first or last value that matches a node -- O(N/2) """
        self._utils.assert_list_not_empty()
        self._validators.enforce_type(element, self.datatype)

        left = self._head
        right = self._tail

        for _ in range((self._total_nodes + 1) // 2):
            if left.element == element:
                return left
            if right.element == element:
                return right

            left = left.next
            right = right.prev

        return None

    def search_all_values(self, element: T) -> Generator["iNode[T]", None, None]:
        """searches for all the Nodes that match a specific value, and yields them as a generator - can be used with loops"""

        self._utils.assert_list_not_empty()
        self._validators.enforce_type(element, self.datatype)

        current_node = self._head

        while current_node:
            # victory condition
            if current_node.element == element:
                yield current_node
            # traverse
            current_node = current_node.next

            # exit condition
            if current_node == self._head:
                break

    def _search_index(self, index: int) -> Optional["iNode[T]"]:
        """ Average O(N/2) -- Adaptive Index Search: Searches for a specific index in the linked list and returns the node or data for further manipulation"""

        self._utils.assert_list_not_empty()
        self._validators.index_boundary_check(index, self._total_nodes)

        # if the index is less than half of the list size - start from the head
        if index < self._total_nodes // 2:
            current_node = self._head
            # loop through to index point
            for _ in range(index):
                current_node = current_node.next
        # otherwise start from the tail:
        else:
            current_node = self._tail
            for _ in range(self._total_nodes - 1, index, -1):
                current_node = current_node.prev
        return current_node

    def search_for_index_by_value(self, element: T, reverse: bool=False) -> Optional[int]:
        """Searches for a specific value and returns the first or last index number (via reverse) for that value."""
        self._utils.assert_list_not_empty()

        index = self._total_nodes -1 if reverse else 0
        step = -1 if reverse else 1
        current_node = self._tail if reverse else self._head
        while current_node:
            if current_node.element == element:
                return index
            index += step
            current_node = current_node.prev if reverse else current_node.next
        return None

    # ----- Mutator Operations -----
    # ------------ insert ------------
    def insert_head(self, element):
        """Insert Node at the head position"""
        new_node = Dll_Node(element, is_linked=True, list_owner=self)

        self._validators.enforce_type(element, self.datatype)
        # Empty Case & 1 Member Case: if the list is empty or has 1 node:
        if self._head is None:
            self._head = self._tail = new_node
            # implement circular logic for head and tail
            self._head.next = self._head.prev = self._tail
            self._tail.next = self._tail.prev = self._head
        # Middle Case: if this the list has multiple items.
        else:
            # new head points to old head
            new_node.next = self._head
            # tail points to new head
            self._tail.next = new_node
            # new head points back to tail (DCL - circular list)
            new_node.prev = self._tail
            # old head points back to new head (now in 2nd pos)
            self._head.prev = new_node
            # new head becomes the main head
            self._head = new_node

        self._total_nodes += 1  # increment size tracker
        return new_node

    def insert_tail(self, element):
        """Insert Node into tail position"""
        new_node = Dll_Node(element, is_linked=True, list_owner=self)

        # if list is empty - or if list has 1 item.
        if self._head is None:
            self._head = self._tail = new_node
            # implement circular logic for head and tail
            self._head.next = self._head.prev = self._tail
            self._tail.next = self._tail.prev = self._head
        # for a list with multiple items
        else:
            # new tail points to head
            new_node.next = self._head
            # old tail points to new tail
            self._tail.next = new_node
            # new tail points back to old tail
            new_node.prev = self._tail
            # head points back to new tail
            self._head.prev = new_node
            # new tail becomes THE tail
            self._tail = new_node
        self._total_nodes += 1  # increment size tracker
        return new_node

    def insert_after(self, node, element):
        """Insert Node in the position after a reference node..."""
        new_node = Dll_Node(element, is_linked=True, list_owner=self)
        # existence check for list
        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)
        self._validators.enforce_type(element, self.datatype)
        # if there is only 1 node (new node becomes the tail)
        if self._head == self._tail:
            # new node points back to head
            new_node.prev = self._head
            # new node points forwards to head (circular)
            new_node.next = self._head
            # head points forwards to new node
            self._head.next = new_node
            # head points back to new node (circular)
            self._head.prev = new_node
            # new node becomes the tail...
            self._tail = new_node
        # if ref node is tail - new node becomes the tail
        elif node == self._tail:
            # new tail points to head
            new_node.next = self._head
            # old tail points to new tail
            self._tail.next = new_node
            # new tail points back to old tail
            new_node.prev = self._tail
            # head points back to new tail
            self._head.prev = new_node
            # new tail becomes THE tail
            self._tail = new_node
        # insert after pos...
        else:
            # new node points back to ref node
            new_node.prev = node
            # new node points forwards to 1 after ref node
            new_node.next = node.next
            # 1 after ref node points back to new node
            node.next.prev = new_node
            # ref node points forwards to new node
            node.next = new_node

        self._total_nodes += 1  # increment tracker
        return new_node

    def insert_before(self, node, element):
        """Inserts a Node before a reference node position"""
        new_node = Dll_Node(element, is_linked=True, list_owner=self)
        # existence check for list
        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)
        self._validators.enforce_type(element, self.datatype)

        # if there is only 1 node or ref node is the head -> (new node becomes the head)
        if self._head == self._tail or node == self._head:
            new_node.next = self._head   # new node points to old head
            new_node.prev = self._tail   # new node points back to tail
            self._head.prev = new_node   # old head points back to new node
            self._tail.next = new_node   # tail points to new node
            self._head = new_node    # new node becomes head
        # insert before pos...
        else:
            new_node.next = node
            new_node.prev = node.prev
            node.prev.next = new_node
            node.prev = new_node
        self._total_nodes += 1
        return new_node

    def replace(self, node, element):
        """replaces a value of a specific node and returns the old value."""
        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)
        self._validators.enforce_type(element, self.datatype)
        old_value = node.element
        node.element = element
        return old_value

    # ------------ delete ------------
    def delete(self, node):
        """Deletes a node from the linked list and returns the old value."""
        # empty list Case:
        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)

        old_node = node
        old_value = node.element
        previous_node = old_node.prev
        future_node = old_node.next

        # 1 member Case: (Head is the tail) - deleting makes the list empty.
        if self._head == self._tail:
            self._head = self._tail = None
        # 2 member Case: deleting a node makes the list a 1 member list.
        elif self._total_nodes == 2:
            remain_node = self._tail if node is self._head else self._head
            remain_node.next = remain_node.prev = remain_node
            self._head = self._tail = remain_node
        else:
            # Main Case:
            previous_node.next = future_node
            future_node.prev = previous_node
            # is head Case:
            if node is self._head:
                self._head = future_node
            # is tail Case:
            if node is self._tail:
                self._tail = previous_node

        # dereference
        old_node.next, old_node.prev = None, None
        old_node.is_linked = False
        old_node.list_owner = None

        self._total_nodes -= 1

        return old_value

    def delete_head(self):
        # is list empty?
        self._utils.assert_list_not_empty()

        old_head = self._head
        old_value = self._head.element    # original head data for return

        # 1 member list Case: if list only has one node.
        if self._head == self._tail:
            self._head = self._tail = None
        # Main Case: if there are multiple nodes - replace head with new head and relink
        else:
            # 1 node after head
            current_node = self._head.next
            current_node.prev = self._tail  # new head points to tail
            self._tail.next = current_node   # tail points to new head
            self._head = current_node    # new head becomes THE head
            # dereference old node
            old_head.prev, old_head.next = None, None
            old_head.is_linked = False
            old_head.list_owner = None

        self._total_nodes -= 1  # decrement tracker
        return old_value

    def delete_tail(self):
        # check if list is not empty
        self._utils.assert_list_not_empty()
        old_tail = self._tail
        old_value = self._tail.element

        # 1 member list Case: if list only 1 node - delete head and tail.
        if self._head == self._tail:
            self._head = self._tail = None
        # Main Case: delete tail only
        else:
            current_node = self._tail.prev   # 1 before tail
            current_node.next = self._head   # new tail point to head
            self._head.prev = current_node   # head point to new tail
            self._tail = current_node    # new tail becomes THE tail
            # dereference
            old_tail.prev, old_tail.next = None, None
            old_tail.is_linked = False
            old_tail.list_owner = None

        self._total_nodes -= 1  # decrement size tracker
        return old_value

    def delete_after(self, node):
        """Delete Node after supplied reference node position..."""
        # check if list is not empty
        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)

        old_node = node.next
        old_value = old_node.element

        # 1 member list Case: if list only has 1 node. delete both head and tail
        if self._head == self._tail:
            raise IndexError("Error: List only has 1 Node. Cannot Delete a Node after...")

        # 2 member list Case: if the list only has 2 nodes... (head & tail)
        elif self._total_nodes == 2:
            #   delete target node. keep the ref node
            remaining_node = node
            self._head = self._tail = remaining_node
            remaining_node.next = remaining_node.prev = remaining_node  # link the final node to itsel
        # Main Case: delete node after ref node
        else:
            node.next = old_node.next    # ref node points to 1 after target node
            old_node.next.prev = node    # 1 after target points back to ref node

        # dereference
        old_node.next = old_node.prev = None  
        old_node.is_linked = False
        old_node.list_owner = None

        self._total_nodes -= 1  # decrement size tracker

        return old_value

    def delete_before(self, node):
        """Delete Node before specified reference node..."""

        self._utils.assert_list_not_empty()
        self._utils.validate_node(node, iNode)

        old_node = node.prev
        old_value = node.prev.data

        # if list is 1 node raise error
        if self._head == self._tail:
            raise IndexError("List only has 1 Node. Cannot Delete a Node before...")

        # if there are only 2 node.
        elif self._total_nodes == 2:
            remain_node = node
            self._head = self._tail = remain_node
            remain_node.next = remain_node.prev = remain_node

        # if there are multiple nodes
        else:
            node.prev = old_node.prev    # ref node points back to 1 before target
            old_node.prev.next = node    # 1 before target points to ref node

        old_node.prev = old_node.next = None # dereference deleted node
        old_node.is_linked = False
        old_node.list_owner = None

        self._total_nodes -= 1 # decrement tracker.

        return old_value

    # ------------ rotate ------------
    def single_rotate_left(self):
        "move head and tail 1 position to the left(forwards)."
        if self._total_nodes == 0:  # empty list
            return

        self._head = self._head.next
        self._tail = self._tail.next

    def rotate_left(self, rotations):
        """We are moving the head and tail left(forwards) by a specific number of rotations"""
        if self._total_nodes == 0:  # empty list
            return

        if rotations % self._total_nodes == 0:  # full cycle rotation - no changes needed
            return

        # the remainder (modulus) of rotations / self.size. ensures rotations never exceeds the length of the list
        rotations %= self._total_nodes

        for _ in range(rotations):
            self._head = self._head.next
            self._tail = self._tail.next

    def single_rotate_right(self):
        "move head and tail 1 position to the right (backwards)."
        if self._total_nodes == 0:  # empty list
            return

        self._head = self._head.prev
        self._tail = self._tail.prev

    def rotate_right(self, rotations):
        """We are moving the head and tail right(backwards) by a specific number of rotations"""
        if self._total_nodes == 0:  # empty list
            return

        if rotations % self._total_nodes == 0:  # full cycle rotation - no changes needed
            return

        # the remainder (modulus) of rotations / self.size. ensures rotations never exceeds the length of the list
        rotations %= self._total_nodes

        for _ in range(rotations):
            self._head = self._head.prev
            self._tail = self._tail.prev


# Main --- Client Facing Code ---
def main():
    # Initialize the list
    print("\n--- Initializing DCLL ---")
    dcll = DoublyCircularList(str)

    print(f"Is empty? {dcll.is_empty()}")

    # ---------- Normal Insertions ----------
    print("\n--- Insertions ---")
    head = dcll.insert_head("0")
    print(dcll)
    node_a = dcll.insert_head("1")
    print(dcll)
    node_b = dcll.insert_tail("2")
    print(dcll)
    node_c = dcll.insert_after(node_b, "3")
    print(dcll)
    node_d = dcll.insert_after(head, "4")    
    print(dcll)
    node_e = dcll.insert_after(node_b, "5")
    print(dcll)
    node_f = dcll.insert_before(node_c, "6")
    print(dcll)
    node_g = dcll.insert_before(node_a, "7")
    print(dcll)

    # ---------- Deletions ----------
    print("\n--- Deletions ---")
    deleted_value = dcll.delete(node_g)
    print(f"Deleted Node: {deleted_value}")
    print(dcll)
    deleted_head = dcll.delete_head()
    print(f"Deleted Head: {deleted_head}")
    print(dcll)
    deleted_tail = dcll.delete_tail()
    print(f"Deleted Tail: {deleted_tail}")
    print(dcll)

    # ---------- Iteration ----------
    print("\n--- Iteration ---")
    for i, item in enumerate(dcll):
        print(f"Iterated over: {i}: {item}")

    # ---------- Searches ----------
    print("\n--- Searches ---")
    print(repr(dcll.search_value("5")))
    print(repr(dcll.search_value("0", reverse=True)))
    print(dcll.search_for_index_by_value("4"))
    print(dcll.bidirectional_search_value("2"))

    # ---------- Get/Set Item ----------
    print("\n--- Get/Set Item ---")
    print(f"Get item at index 1: {dcll[1]}")
    dcll[1] = "565"
    print(f"Set item at index 1: {dcll[1]}")

    print(f"Get item via node reference: {dcll[node_f]}")
    dcll[node_f] = "6849"
    print(f"Set item via node reference: {dcll[node_f]}")
    print(dcll)
    
    print(f"List length: {len(dcll)}")
    print(f"Is '10' in DCLL? {'10' in dcll}")
    print(f"Is '200' in DCLL? {'200' in dcll}")

    # ---------- Clear ----------
    print("\n--- Clearing List ---")
    dcll.clear()
    print(dcll)
    print(f"Is empty after clear? {dcll.is_empty()}")

    print("\n\n--- Testing Error Cases ---")
    dcll = DoublyCircularList(str)
    node = dcll.insert_head("A")

    # enforce_type errors
    try:
        dcll.insert_head(RandomClass("YOLO"))
    except Exception as e:
        print(f"Caught enforce_type error: {e}")

    try:
        dcll.replace(node, RandomClass("YOLO"))
    except Exception as e:
        print(f"Caught enforce_type error: {e}")

    # validate_node errors
    fake_node = "not_a_node"
    try:
        dcll.insert_after(fake_node, "X")
    except Exception as e:
        print(f"Caught validate_node error: {e}")

    try:
        dcll.delete(fake_node)
    except Exception as e:
        print(f"Caught validate_node error: {e}")

    # deleting from empty list
    try:
        dcll.clear()
    except Exception as e:
        print(f"Caught Error: {e}")

    try:
        dcll.delete_head()
    except Exception as e:
        print(f"Caught empty list error: {e}")

    try:
        dcll.delete_tail()
    except Exception as e:
        print(f"Caught empty list error: {e}")

if __name__ == "__main__":
    main()
