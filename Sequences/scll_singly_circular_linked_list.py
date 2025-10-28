from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator
from abc import ABC, ABCMeta, abstractmethod


"""
Circular Linked List: The Tail - points back to the head
Most libraries implement CLLs with singly circular links; doubly circular is less common but more versatile.
"""

T = TypeVar('T')

class iNode(ABC, Generic[T]):

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Node(iNode[T]):
    def __init__(self, data) -> None:
        self.data = data
        self.next: Optional[Node[T]] = None

    def __repr__(self) -> str:
        return f"Node: {self.data}"


class iCircularLinkedList(ABC, Generic[T]):

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
    def traverse(self, function: Callable[[T], Any]) -> list[T]:
        pass

    # ------------ search ------------
    @abstractmethod
    def search_value(self, value: T, return_node: bool) -> "Optional[Node[T] | T]":
        pass

    @abstractmethod
    def search_all_values(self, value: T, return_node: bool) -> list[T]:
        pass

    @abstractmethod
    def _search_index(self, index: int, return_node: bool) -> "Optional[Node[T] | T]":
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
    def insert_at(self, value: T, index: int):
        pass

    # ------------ delete ------------
    @abstractmethod
    def delete_head(self) -> T:
        pass

    @abstractmethod
    def delete_tail(self) -> T:
        pass

    @abstractmethod
    def delete_at(self, index: int) -> T:
        pass


class CircularLinkedList(iCircularLinkedList[T]):
    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None
        self.size: int = 0

    # ------------ Utility ------------
    def _boundary_check(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index Out Of Bounds...")

    def _empty_list(self):
        if self.head is None:
            raise IndexError("List is Empty")

    def __iter__(self) -> Iterator[T]:
        current_node = self.head
        for _ in range(self.size):
            yield current_node.data
            current_node = current_node.next

    def __contains__(self, value):
        """ built in override - for boolean contains logic"""
        return self.contains(value)

    def __getitem__(self, index: int) -> "Optional[Node[T] | T]":
        """returns the value of a node in the linked list. Overrides builtin"""
        item = self._search_index(index, return_node=False)
        return item

    def __setitem__(self, index: int, value: T) -> None:
        """sets the value of a node in the linked list. Overrides builtin:"""
        node = self._search_index(index, return_node=True)
        if node:
            node.data = value

    def __str__(self) -> str:
        """All the contents of the list"""
        results = []
        if self.head is None:
            return f"List is Empty"
        current_node = self.head
        for _ in range(self.size):
            current_node = current_node.next
            results.append(current_node.data)
        nodes = ", ".join(results)
        return nodes

    def clear(self):
        """runs delete head repeatedly until there are no nodes left"""
        current_node = self.head
        for _ in range(self.size):
            self.delete_head()

    def length(self):
        """returns how many nodes in the list there are"""
        return self.size

    def is_empty(self):
        """Checks if the List is empty"""
        return self.head is None

    def contains(self, value):
        """Does the linked list contain this value?"""
        current_node = self.head
        for _ in range(self.size):
            if value == current_node.data:
                return True
            current_node = current_node.next
        return False

    # ------------ Traverse ------------
    def traverse(self, function):
        """ Traverse List and apply function. Store results in a list and return them"""
        results = []
        self._empty_list()
        current_node = self.head
        for _ in range(self.size):
            transformation = function(current_node.data)
            results.append(transformation)
            current_node = current_node.next
        return results

    # ------------ search ------------
    def search_value(self, value, return_node):
        """Searches for a value in the list and returns the first match"""
        self._empty_list()
        current_node = self.head
        for _ in range(self.size):
            if current_node.data == value:
                return current_node if return_node else current_node.data
            current_node = current_node.next
        return None

    def search_all_values(self, value, return_node):
        """Searches through all nodes, and collects values that match in a list"""
        self._empty_list()
        results = []
        current_node = self.head
        for _ in range(self.size):
            if current_node.data == value:
                results.append(current_node if return_node else current_node.data)
            current_node = current_node.next
        return results

    def _search_index(self, index, return_node):
        """searches for a node by index"""
        self._empty_list()
        self._boundary_check(index)
        current_node = self.head
        for _ in range(index):
            current_node = current_node.next
        return current_node if return_node else current_node.data

    def search_for_index_by_value(self, value):
        """returns the index of the first matched value in the index."""
        self._empty_list()
        current_node = self.head
        # traverse list - on match return index number
        for index in range(self.size):
            if current_node.data == value:
                return index
            current_node = current_node.next
        return None # value not found

    # ------------ insert ------------
    def insert_head(self, value):
        """Inserts a Node at the head. O(1)"""
        new_head = Node(value)
        # are there other nodes apart from the head and the tail?

        # if there is no head, the new node becomes the head and the tail.
        if self.head is None:
            self.head = new_head
            self.tail = self.head
            # tail links back to head
            self.tail.next = self.head

        else:
            # point new node to current head -- [new_head] > [old_head] > [tail]
            new_head.next = self.head
            # point tail to the new head -- [new_head] > [old_head] > [tail] > [new_head]
            self.tail.next = new_head
            # assign head to the new node  [head] > [node] > [tail] > [head]
            self.head = new_head

        self.size += 1  # update size tracker

    def insert_tail(self, value):
        """Inserts a node ath the tail - O(1)"""
        new_node = Node(value)
        # is list empty? - tail becomes head
        if self.head is None:
            self.head = new_node
            self.tail = self.head
            self.tail.next = self.head

        # otherwise - insert new node after tail
        else:
            # link old tail to new node
            self.tail.next = new_node
            # insert at tail
            self.tail = new_node
            # link new tail to head
            self.tail.next = self.head

        # increment size counter
        self.size += 1

    def insert_at(self, value, index):
        """Insert Node at index position - O(N)"""
        new_node = Node(value)
        current_node = self.head
        # if index value = head - insert head() is O(1)
        if index <= 0:
            self.insert_head(value)
            return
        # if index value is the last position of the list - insert tail() O(1)
        elif index >= self.size:
            self.insert_tail(value)
            return

        # move to index position
        for _ in range(index-1):
            current_node = current_node.next

        # new node points to future node
        new_node.next = current_node.next
        # previous node points to new node
        current_node.next = new_node

        self.size += 1  # update size tracker

    # ------------ delete ------------
    def delete_head(self):
        # does head exist?
        self._empty_list()

        old_head = self.head

        # single node list - delete both head and tail
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next  # head.next becomes the new head
            self.tail.next = self.head  # connect tail to new head

        old_head.next = None
        self.size -= 1
        return old_head.data

    def delete_tail(self):
        """Delete Node at Tail"""
        # does head exist?
        self._empty_list()

        # single node list - head & tail are the same. run delete head()
        if self.head == self.tail:
            return self.delete_head()
        else:
            # initialize search
            old_tail = self.tail
            current_node = self.head
            # traverse to 1 before tail
            for _ in range(self.size -2):
                current_node = current_node.next
            # current node point to head
            current_node.next = self.head
            # current node becomes tail
            self.tail = current_node
            old_tail.next = None    # dereference old tail
            self.size -= 1  # decrement tracker
            return old_tail.data

    def delete_at(self, index):
        """Delete Node at specified index"""
        # empty list - throw error
        self._empty_list()
        # only 1 item - delete head - O(1)
        if index <= 0:
            return self.delete_head()
        # index is the tail - delete tail - O(1)
        elif index >= self.size - 1:
            return self.delete_tail()
        # initialize node
        current_node = self.head
        # travel to 1 before index (previous node)
        for _ in range(index-1):
            current_node = current_node.next
        # target node for deletion (target node)
        target_node = current_node.next
        # link previous node to the node after current node
        current_node.next = target_node.next
        target_node.next = None    # dereference old node
        self.size -= 1  # decrement tracker
        return target_node.data


# Main --- Client Facing Code ----
def main():
    cll = CircularLinkedList[int]()

    print("=== Empty list checks ===")
    print(f"Is empty: {cll.is_empty()} -- Length: {cll.length()}")

    print("\n=== Insert head and tail ===")
    cll.insert_head(1)
    cll.insert_tail(2)
    cll.insert_head(0)
    cll.insert_tail(3)
    print(f"List after inserts: {list(cll)}")
    print(f"Head: {cll.head.data}, Tail: {cll.tail.data}")
    print(f"Tail next (should point to head): {cll.tail.next.data}")

    print("\n=== Insert at index ===")
    cll.insert_at(99, 2)
    cll.insert_at(100, 0)  # head
    cll.insert_at(101, 100)  # tail
    print(f"List after insert_at: {list(cll)}")

    print("\n=== Delete head, tail, at index ===")
    print(f"Deleted head: {cll.delete_head()}")
    print(f"Deleted tail: {cll.delete_tail()}")
    print(f"Deleted at index 2: {cll.delete_at(2)}")
    print(f"List after deletions: {list(cll)}")

    print("\n=== Search operations ===")
    cll.insert_tail(2)
    cll.insert_tail(3)
    cll.insert_tail(2)
    print(f"List: {list(cll)}")
    print(f"Contains 2: {cll.contains(2)}")
    print(f"Contains 99: {cll.contains(99)}")
    print(f"Index of 2: {cll.search_for_index_by_value(2)}")
    print(f"Search value (first 2) node: {cll.search_value(2, return_node=True)}")
    print(f"Search value (first 2) data: {cll.search_value(2, return_node=False)}")
    print(f"All values 2: {cll.search_all_values(2, return_node=False)}")
    print(f"All nodes 2: {cll.search_all_values(2, return_node=True)}")

    print("\n=== Traverse ===")
    print(f"Traverse *2: {cll.traverse(lambda x: x*2)}")

    print("\n=== Iteration ===")
    for val in cll:
        print(f"Iterated value: {val}")

    print("\n=== Clear list ===")
    cll.clear()
    print(f"After clear, is empty: {cll.is_empty()}")
    print(f"Length after clear: {cll.length()}")



if __name__ == "__main__":
    main()
