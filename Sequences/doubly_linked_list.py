from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator
from abc import ABC, ABCMeta, abstractmethod


"""
A Doubly Linked List (DLL) has both a previous and a next pointer.
Can move forwards or backwards in the list. 
"""

T = TypeVar('T')

# interfaces
class iDoublyLinkedList(ABC, Generic[T]):

    # ------------ Utility ------------
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def contains(self, value) -> bool:
        pass

    # ------------ Traverse ------------
    @abstractmethod
    def traverse(self, function: Callable, start_from_tail: bool) -> list[T]:
        pass

    # ------------ search ------------
    @abstractmethod
    def search_value(self, value: T, return_node: bool) -> 'Optional[Node[T] | T]' :
        pass

    @abstractmethod
    def search_all_values(self, value: T, return_node: bool) -> Optional[list[T]]:
        pass

    @abstractmethod
    def search_index(self, index: int, return_node: bool) -> "Optional[Node[T] | T]":
        pass

    @abstractmethod
    def search_for_index_by_value(self, value: T) -> Optional[int]:
        pass

    # ------------ insert ------------
    @abstractmethod
    def insert_head(self, value: T):
        pass

    @abstractmethod
    def insert_tail(self, value: T):
        pass

    @abstractmethod
    def insert_after(self, node: 'Node[T]', value: T):
        pass

    @abstractmethod
    def insert_before(self, node: 'Node[T]', value: T):
        pass

    # ------------ delete ------------
    @abstractmethod
    def delete_head(self) -> T:
        pass

    @abstractmethod
    def delete_tail(self) -> T:
        pass

    @abstractmethod
    def delete_after(self, node: 'Node[T]') -> T:
        pass

    @abstractmethod
    def delete_before(self, node: 'Node[T]') -> T:
        pass


# Node
class iNode(ABC, Generic[T]):

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Node(iNode[T]):
    def __init__(self, data: T) -> None:
        self.data: T = data
        self.prev: Optional[Node[T]] = None
        self.next: Optional[Node[T]] = None

    def __repr__(self) -> str:
        return f"Node: {self.data}"


# Double Linked List
class DoublyLinkedList(iDoublyLinkedList[T]):
    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None
        self.size: int = 0

    # ------------ Utilities ------------
    def _node_exists(self, node: Optional[Node[T]]):
        if node is None:
            raise ValueError("Node cannot be None, please give a valid Node.")

    def _list_exists(self):
        if not self.head:
            raise IndexError("List is Empty...")

    def _index_boundary_check(self, index: int):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range...")

    def __iter__(self):
        """Builtin: List can now be iterated over in loops etc..."""
        current_node = self.head
        while current_node:
            yield current_node.data
            current_node = current_node.next

    def __reversed__(self):
        """Python Built in - reverses iteration"""
        current_node = self.tail
        while current_node:
            yield current_node.data
            current_node = current_node.prev

    def __len__(self) -> int:
        """Override Python Built in to ensure the nodes list is returned"""
        return self.size

    def __contains__(self, value):
        """Overrides python built in to ensure - custom logic for evaluating the whole list by value is implemented"""
        return self.contains(value)

    def __getitem__(self, index: int) -> "Optional[Node[T] | T]":
        """returns the value of a node in the linked list. Overrides builtin"""
        item = self.search_index(index, return_node=False)
        return item

    def __setitem__(self, index: int, value: T) -> None:
        """sets the value of a node in the linked list. Overrides builtin:"""
        node = self.search_index(index, return_node=True)
        if node:
            node.data = value

    def validate_list(self) -> None:
        """Helper Method - Validates that all the links in the list are still connected to each other, and that the size of the list is accurate."""
        self._list_exists()

        current_node = self.head
        count = 0

        while current_node:
            if current_node.next and current_node.next.prev != current_node:
                raise RuntimeError("Broken link in Doubly Linked List Detected!")
            count +=1
            current_node = current_node.next
        if count != self.size:
            raise RuntimeError("List size does not match counted nodes!")

    # ------------ Public Methods ------------

    # ------------ General ------------

    def clear(self):
        """Deletes all items from the linked list"""
        while not self.is_empty():
            self.delete_head()

    def length(self):
        """returns the Number of nodes in the linked list"""
        return self.size

    def is_empty(self):
        """Boolean Check if the list is empty"""
        return self.head is None

    def contains(self, value):
        """return True or False if a node contains the specified data."""
        current_node = self.head
        while current_node:
            if current_node.data == value:
                return True
            current_node = current_node.next
        return False

    # ------------ Insertions ------------

    def insert_head(self, value):
        """add a new node at the very beginning of the list — making it the new head."""
        new_node = Node(value)
        # Old head is now after new node. (if list is empty = None)
        new_node.next = self.head 

        # update old head prev pointer: (points to the current head - new node)
        if self.head:
            self.head.prev = new_node  
        # Empty list (head and tail are the same Node)
        else:
            self.tail = new_node
        # Assign New node to the head
        self.head = new_node
        # increment size tracker
        self.size += 1

    def insert_tail(self, value):
        """insert a node at the end of the list - the tail."""
        new_node = Node(value)

        # new tail prev should point to old tail.
        new_node.prev = self.tail

        # is there an exsiting tail. (if so should point to new tail)
        if self.tail:
            self.tail.next = new_node
        # if list is empty - head and tail are the same so insert at head and tail
        else:
            self.head = new_node
        # insert new node at tail
        self.tail = new_node
        # increment size tracker
        self.size += 1

    def insert_after(self, node, value):
        """Inserts a new node after a specific node -- O(1)"""
        self._node_exists(node) # existence check
        new_node = Node(value)  # create node
        # Step 1: link the new node to the previous node
        new_node.prev = node
        # Step 2: link the new node to the future node
        new_node.next = node.next
        # Step 3: link the future node to the new node
        if node.next:
            node.next.prev = new_node
        # if the future node doesnt exist. - assign to tail(end)
        else:
            self.tail = new_node
        # Step 4: link the previous node to the new node
        node.next = new_node
        self.size += 1  # increment size tracker

    def insert_before(self, node, value):
        """Inserts a new node before a specific node -- O(1)"""
        self._node_exists(node)  # existence check
        new_node = Node(value)  # create node
        # Step 1: link the new node to the future node
        new_node.next = node
        # Step 2: link the new node to the previous node
        new_node.prev = node.prev
        # Step 3: link the previous node to the new node
        if node.prev:
            node.prev.next = new_node
        # if there is no previous node - assign to the head(start)
        else:
            self.head = new_node
        # Step 4: link the future node to the new node
        node.prev = new_node
        self.size += 1  # increment size tracker

    # ------------ Deletions ------------

    def delete_head(self):
        """Deletes Node at the Head Position"""
        self._list_exists() # existence check for head
        removed_node = self.head.data
        old_head = self.head
        # if there is only 1 node:
        if self.head == self.tail:
            self.head = self.tail = None
        # Otherwise - move head forward 1 node and delete its previous link
        else:
            self.head = old_head.next
            self.head.prev = None
            # dereferencing old head
            old_head.prev, old_head.next = None, None
        self.size -= 1  # update size tracker
        return removed_node

    def delete_tail(self):
        """Deletes the Node at the Tail Position"""
        self._list_exists() # existence check
        old_tail = self.tail    # for dereferencing
        removed_value = self.tail.data
        # if there is only 1 node (head & tail)
        if self.head == self.tail:
            self.head = self.tail = None
        # if the tail exists, dereference and set future node to none
        else:
            self.tail = old_tail.prev
            self.tail.next = None   # tail future node must always be None
            # fully detach the old tail from the list
            old_tail.prev, old_tail.next = None, None
        self.size -= 1  # decrement size counter
        return removed_value

    def delete_after(self, node):
        """Deletes Node that comes after a specified Node"""

        self._list_exists() # existence check

        # checks if there is a node after the specified node
        if not node.next:
            raise IndexError("No node exists after the specified node...")

        # Step 1: Set Target for deletion
        target_node = node.next
        deleted_node_data = node.next.data
        # Step 2: Update Node to point to future node (the node after the deleted node)
        node.next = target_node.next
        # Step 3: if there is a future node (after deleted node): link that to Node
        if target_node.next:
            target_node.next.prev = node
        # Step 3B: otherwise - the deleted node was the Tail - so update Node to become the tail
        else:
            self.tail = node
        # Step 4: dereference deleted node
        target_node.next, target_node.prev = None, None

        self.size -= 1  # decrement size tracker
        return deleted_node_data

    def delete_before(self, node):
        """Deletes Node that comes before a specified Node"""

        self._list_exists() # existence check

        if not node.prev:
            raise IndexError("No Node exists before the specified node...")

        # Step 1: set Target for deletion
        target_node = node.prev
        target_node_data = target_node.data

        # Step 2: if there is a node before the target, assign its future node to ref node, and reassign ref node's previous node to the 1 before the target
        if target_node.prev:
            target_node.prev.next = node
            node.prev = target_node.prev
        # otherwise target node is the head - change to node
        else:
            self.head = node
            node.prev = None    # dereferences any prior links

        # Dereference Deleted Node
        target_node.prev, target_node.next = None, None
        self.size -= 1
        return target_node_data

    # ------------ Searches ------------

    def search_value(self, value, return_node=True):
        """
        Bidirectional Search: Average O(N/2), worst O(N) Return the first node containing the value, or None if not found.
        Bidirectional traversal improves latency for early exits, not throughput for full scans.
        """
        if self.head is None:
            return None
        # initialize starter nodes
        left = self.head
        right = self.tail
        # Existence check and crossover check
        while (left and right) and (left != right.next):
            if left.data == value:
                return left if return_node else left.data
            if right.data == value:
                return right if return_node else right.data
            # move to next step
            left = left.next
            right = right.prev

        return None # No value found

    def search_all_values(self, value, return_node=True):
        """return all nodes (as a list) that contain a value or None if not found..."""
        results = []
        current_node = self.head

        while current_node:
            if current_node.data == value:
                results.append(current_node if return_node else current_node.data)
            current_node = current_node.next

        return results

    def search_for_index_by_value(self, value):
        """Return the index of the first node with the value, or None if not found."""
        current_node = self.head
        index = 0
        while current_node:
            if current_node.data == value:
                return index
            index += 1
            current_node = current_node.next
        return None

    def search_index(self, index, return_node=True):
        """ Average O(N/2) -- Adaptive Index Search: Searches for a specific index in the linked list and returns the node or data for further manipulation"""
        self._index_boundary_check(index)
        self._list_exists()

        # if the index is less than half of the list size - start from the head
        if index < self.size // 2:
            current_node = self.head
            if not current_node:
                raise IndexError("List is Empty...")
            # loop through to index point
            for _ in range(index):
                current_node = current_node.next
        # otherwise start from the tail:
        else:
            current_node = self.tail
            for _ in range(self.size -1, index, -1):
                current_node = current_node.prev
        return current_node if return_node else current_node.data

    # ------------ Searches ------------

    def traverse(self, function, start_from_tail=False):
        """Apply a function to each element and return a new list of results. Can traverse forwards or backwards"""
        results = []
        current_node = self.tail if start_from_tail else self.head
        while current_node:
            transformation = function(current_node.data)
            results.append(transformation)
            current_node = current_node.prev if start_from_tail else current_node.next
        return results
