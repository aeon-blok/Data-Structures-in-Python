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
)
from abc import ABC, ABCMeta, abstractmethod


"""Doubly Circular Linked List - where the tail reconnects to the head."""

# TODO: Add delete by value
# TODO: add Slice
# TODO: Merging Linked lists (maintaining order)
# TODO: Splitting linked lists
# TODO: add Search all indices (returns multiple indices...)
# TODO: Conditional Travere
# TODO: Count how many times a value appears in the linked list...
# TODO: full list reversal - and partial list reversal
# TODO: remove every N node
# TODO: Swap nodes in pairs....

T = TypeVar('T')


class iNode(ABC, Generic[T]):
    """Interface for Node"""

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Node(iNode[T]):
    """Node Implementation"""
    def __init__(self, data: T) -> None:
        self.data = data
        self.next: Optional[Node[T]] = None
        self.prev: Optional[Node[T]] = None

    def __repr__(self) -> str:
        return f"Node: {self.data}"


class iDoublyCircularList(ABC, Generic[T]):
    """Interface for DCL"""

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
    def traverse(
        self, function: Callable[[T], Any]
    ) -> Generator[Node[T] | T, None, None]:
        pass

    # ------------ search ------------
    @abstractmethod
    def bidirectional_search_value(self, value: T, return_node: bool) -> "Optional[Node[T] | T]":
        pass

    @abstractmethod
    def search_value(self, value: T, return_node: bool) -> "Optional[Node[T] | T]":
        pass

    @abstractmethod
    def search_all_values(
        self, value: T, return_node: bool
    ) -> Generator[Node[T] | T, None, None]:
        pass

    @abstractmethod
    def _search_index(self, index: int) -> Optional[Node[T]]:
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
    def insert_after(self, value: T, ref_node: Node[T]):
        pass

    @abstractmethod
    def insert_before(self, value: T, ref_node: Node[T]):
        pass

    # ------------ delete ------------
    @abstractmethod
    def delete_head(self) -> Optional[T]:
        pass

    @abstractmethod
    def delete_tail(self) -> Optional[T]:
        pass

    @abstractmethod
    def delete_before(self, ref_node: Node[T]) -> Optional[T]:
        pass

    @abstractmethod
    def delete_after(self, ref_node: Node[T]) -> Optional[T]:
        pass

class DoublyCircularList(iDoublyCircularList[T]):
    """Implementation of DCL: """
    def __init__(self) -> None:
        self.head = None
        self.tail = None
        self.size = 0

    # ------------ Exceptions ------------

    def _list_exists(self):
        if self.head is None:
            raise IndexError("List is empty...")

    def _boundary_check(self, index):
        """checks if the index value is within the size of the list."""
        if index < 0 or index >= self.size:
            raise IndexError("Index Out Of Bounds...")

    # ------------ Built in Overrides ------------
    def __iter__(self) -> Iterator[T]:
        current_node = self.head
        while current_node:
            yield current_node.data
            current_node = current_node.next
            if current_node == self.head:
                break

    def __getitem__(self, index: int):
        """returns the value of a node in the linked list. Overrides builtin -- array like indexing with a linked list but its O(N)"""
        node = self._search_index(index)
        return node.data

    def __setitem__(self, index: int, value: T):
        """Overrides python built in - access linked list like an array -- O(N)"""
        node = self._search_index(index)
        if node:
            node.data = value

    def __contains__(self, value: T):
        """Override Python Built in with custom contains boolean logic."""
        return self.contains(value)

    def __reversed__(self):
        """Python Built in - reverses iteration"""
        current_node = self.tail
        while current_node:
            yield current_node.data
            current_node = current_node.prev
            if current_node == self.tail:
                break

    def __len__(self):
        return self.length()

    def __str__(self) -> str:
        """Displays all the content of the linked list as a string."""

        seperator = " ->> "

        if self.head is None:
            return f"List is Empty"

        def _simple_traversal():
            """traverses the nodes and returns a string via generator"""
            current_node = self.head
            while current_node:
                yield str(current_node.data)
                current_node = current_node.next
                if current_node == self.head:
                    break

        infostring = f"[head]{seperator.join(_simple_traversal())}[tail]"

        return infostring

    # ------------ Utility ------------
    def clear(self):
        """removes all items from the list."""
        while self.size > 0:
            self.delete_head()

    def length(self) -> int:
        return self.size

    def is_empty(self) -> bool:
        return self.head is None

    def contains(self, value) -> bool:
        """Does the Linked list contain this value? utilizes bidirectional search O(N/2)"""
        left = self.head
        right = self.tail

        for _ in range((self.size + 1) // 2): # half traversal only
            if left.data == value or right.data == value:
                return True
            left = left.next
            right = right.prev
        return False

    # ------------ Traverse ------------
    def traverse(self, function):
        """Traverse List and apply function. yield result as a generator for easy parsing with loops"""

        self._list_exists()

        current_node = self.head

        while current_node:
            try:
                yield function(current_node.data)
            except Exception as error:
                print(
                    f"There was an error while trying to apply function to the node: {current_node.data}: {error}"
                )
            finally:
                current_node = current_node.next
                if current_node == self.head:
                    break  

    # ------------ search ------------
    def search_value(self, value, return_node, reverse=False):
        """Finds the first value that matches a node -- O(N)"""
        self._list_exists()
        current_node = self.tail if reverse else self.head
        while current_node:
            if current_node.data == value:
                return current_node if return_node else current_node.data
            current_node = current_node.prev if reverse else current_node.next
            if current_node == (self.tail if reverse else self.head): 
                break
        return None

    def bidirectional_search_value(self, value, return_node):
        """Searches for the first or last value that matches a node -- O(N/2) """
        self._list_exists()

        left = self.head
        right = self.tail

        for _ in range((self.size + 1) // 2):
            if left.data == value:
                return left if return_node else left.data
            if right.data == value:
                return right if return_node else right.data
            left = left.next
            right = right.prev

        return None

    def search_all_values(self, value, return_node):
        """searches for all the Nodes that match a specific value, and yields them as a generator - can be used with loops"""
        self._list_exists()

        current_node = self.head

        while current_node:
            if current_node.data == value:
                yield current_node if return_node else current_node.data
            current_node = current_node.next
            if current_node == self.head:
                break

    def _search_index(self, index):
        """ Average O(N/2) -- Adaptive Index Search: Searches for a specific index in the linked list and returns the node or data for further manipulation"""

        self._list_exists()
        self._boundary_check(index)

        # if the index is less than half of the list size - start from the head
        if index < self.size // 2:
            current_node = self.head
            # loop through to index point
            for _ in range(index):
                current_node = current_node.next
        # otherwise start from the tail:
        else:
            current_node = self.tail
            for _ in range(self.size - 1, index, -1):
                current_node = current_node.prev
        return current_node

    def search_for_index_by_value(self, value, reverse=False):
        """Searches for a specific value and returns the first or last index number (via reverse) for that value."""
        self._list_exists()
        index = self.size -1 if reverse else 0
        step = -1 if reverse else 1
        current_node = self.tail if reverse else self.head
        while current_node:
            if current_node.data == value:
                return index
            index += step
            current_node = current_node.prev if reverse else current_node.next
        return None

    # ------------ rotate ------------

    def single_rotate_left(self):
        "move head and tail 1 position to the left(forwards)."
        if self.size == 0:  # empty list
            return

        self.head = self.head.next
        self.tail = self.tail.next

    def rotate_left(self, rotations):
        """We are moving the head and tail left(forwards) by a specific number of rotations"""
        if self.size == 0:  # empty list
            return

        if rotations % self.size == 0:  # full cycle rotation - no changes needed
            return

        # the remainder (modulus) of rotations / self.size. ensures rotations never exceeds the length of the list
        rotations %= self.size

        for _ in range(rotations):
            self.head = self.head.next
            self.tail = self.tail.next

    def single_rotate_right(self):
        "move head and tail 1 position to the right (backwards)."
        if self.size == 0:  # empty list
            return

        self.head = self.head.prev
        self.tail = self.tail.prev

    def rotate_right(self, rotations):
        """We are moving the head and tail right(backwards) by a specific number of rotations"""
        if self.size == 0:  # empty list
            return

        if rotations % self.size == 0:  # full cycle rotation - no changes needed
            return

        # the remainder (modulus) of rotations / self.size. ensures rotations never exceeds the length of the list
        rotations %= self.size

        for _ in range(rotations):
            self.head = self.head.prev
            self.tail = self.tail.prev

    # ------------ insert ------------
    def insert_head(self, value):
        """Insert Node at the head position"""
        new_node = Node(value)
        # if the list is empty or has 1 node:
        if self.head is None:
            self.head = self.tail = new_node
            # implement circular logic for head and tail
            self.head.next = self.head.prev = self.tail
            self.tail.next = self.tail.prev = self.head
        # if this the list has multiple items.
        else:
            # new head points to old head
            new_node.next = self.head
            # tail points to new head
            self.tail.next = new_node
            # new head points back to tail (DCL - circular list)
            new_node.prev = self.tail
            # old head points back to new head (now in 2nd pos)
            self.head.prev = new_node
            # new head becomes the main head
            self.head = new_node

        self.size += 1  # increment size tracker

    def insert_tail(self, value):
        """Insert Node into tail position"""
        new_node = Node(value)

        # if list is empty - or if list has 1 item.
        if self.head is None:
            self.head = self.tail = new_node
            # implement circular logic for head and tail
            self.head.next = self.head.prev = self.tail
            self.tail.next = self.tail.prev = self.head
        # for a list with multiple items
        else:
            # new tail points to head
            new_node.next = self.head
            # old tail points to new tail
            self.tail.next = new_node
            # new tail points back to old tail
            new_node.prev = self.tail
            # head points back to new tail
            self.head.prev = new_node
            # new tail becomes THE tail
            self.tail = new_node
        self.size += 1  # increment size tracker

    def insert_after(self, value, ref_node):
        """Insert Node in the position after a reference node..."""
        new_node = Node(value)
        # existence check for list
        self._list_exists()
        # if there is only 1 node (new node becomes the tail)
        if self.head == self.tail:
            # new node points back to head
            new_node.prev = self.head
            # new node points forwards to head (circular)
            new_node.next = self.head
            # head points forwards to new node
            self.head.next = new_node
            # head points back to new node (circular)
            self.head.prev = new_node
            # new node becomes the tail...
            self.tail = new_node
        # if ref node is tail - new node becomes the tail
        elif ref_node == self.tail:
            # new tail points to head
            new_node.next = self.head
            # old tail points to new tail
            self.tail.next = new_node
            # new tail points back to old tail
            new_node.prev = self.tail
            # head points back to new tail
            self.head.prev = new_node
            # new tail becomes THE tail
            self.tail = new_node
        # insert after pos...
        else:
            # new node points back to ref node
            new_node.prev = ref_node
            # new node points forwards to 1 after ref node
            new_node.next = ref_node.next
            # 1 after ref node points back to new node
            ref_node.next.prev = new_node
            # ref node points forwards to new node
            ref_node.next = new_node
        self.size += 1  # increment tracker

    def insert_before(self, value, ref_node):
        """Inserts a Node before a reference node position"""
        new_node = Node(value)
        # existence check for list
        self._list_exists()
        # if there is only 1 node or ref node is the head -> (new node becomes the head)
        if self.head == self.tail or ref_node == self.head:
            new_node.next = self.head   # new node points to old head
            new_node.prev = self.tail   # new node points back to tail
            self.head.prev = new_node   # old head points back to new node
            self.tail.next = new_node   # tail points to new node
            self.head = new_node    # new node becomes head
        # insert before pos...
        else:
            new_node.next = ref_node
            new_node.prev = ref_node.prev
            ref_node.prev.next = new_node
            ref_node.prev = new_node
        self.size += 1

    # ------------ delete ------------
    def delete_head(self):
        # is list empty?
        self._list_exists()

        removed_node = self.head.data    # original head data for return

        # if list only has one node.
        if self.head == self.tail:
            self.head = self.tail = None
        # if there are multiple nodes - replace head with new head and relink
        else:
            # 1 node after head
            current_node = self.head.next
            current_node.prev = self.tail  # new head points to tail
            self.tail.next = current_node   # tail points to new head
            self.head = current_node    # new head becomes THE head
        self.size -= 1  # decrement tracker
        return removed_node

    def delete_tail(self):
        # check if list is not empty
        self._list_exists()
        removed_node = self.tail.data
        # if list only 1 node - delete head and tail.
        if self.head == self.tail:
            self.head = self.tail = None
        # delete tail only
        else:
            current_node = self.tail.prev   # 1 before tail
            current_node.next = self.head   # new tail point to head
            self.head.prev = current_node   # head point to new tail
            self.tail = current_node    # new tail becomes THE tail
        self.size -= 1  # decrement size tracker
        return removed_node

    def delete_after(self, ref_node):
        """Delete Node after supplied reference node position..."""
        # check if list is not empty
        self._list_exists()
        removed_node = ref_node.next.data
        target_node = ref_node.next
        # if list only has 1 node. delete both head and tail
        if self.head == self.tail:
            raise IndexError("List only has 1 Node. Cannot Delete a Node after...")
        # if the list only has 2 nodes... (head & tail)
        elif self.size == 2:
            #   delete target node. keep the ref node
            remaining_node = ref_node
            self.head = self.tail = remaining_node
            remaining_node.next = remaining_node.prev = remaining_node  # link the final node to itsel
            target_node.next = target_node.prev = None # dereference deleted node
            del target_node
        # delete node after ref node
        else:
            ref_node.next = target_node.next    # ref node points to 1 after target node
            target_node.next.prev = ref_node    # 1 after target points back to ref node
            target_node.next = target_node.prev = None  # dereference
            del target_node

        self.size -= 1  # decrement size tracker

        return removed_node

    def delete_before(self, ref_node):
        """Delete Node before specified reference node..."""
        self._list_exists() # existence check
        removed_node = ref_node.prev.data
        target_node = ref_node.prev

        # if list is 1 node raise error
        if self.head == self.tail:
            raise IndexError("List only has 1 Node. Cannot Delete a Node before...")
        # if there are only 2 node.
        elif self.size == 2:
            remain_node = ref_node
            self.head = self.tail = remain_node
            remain_node.next = remain_node.prev = remain_node
            target_node.next = target_node.prev = None  # dereference target node.
            del target_node
        # if there are multiple nodes
        else:
            ref_node.prev = target_node.prev    # ref node points back to 1 before target
            target_node.prev.next = ref_node    # 1 before target points to ref node
            target_node.prev = target_node.next = None # dereference deleted node
            del target_node
        self.size -= 1 # decrement tracker.

        return removed_node


# Main --- Client Facing Code ---
def main():
    # Initialize the list
    dcl = DoublyCircularList[int]()
    print(dcl.is_empty())        # True
    print(dcl.length())          # 0
    print(dcl)                   # List is Empty

    # Insert head
    dcl.insert_head(10)
    print(dcl)                   # [head]10[tail]
    print(dcl.is_empty())        # False
    print(dcl.length())          # 1

    # Insert tail
    dcl.insert_tail(20)
    print(dcl)                   # [head]10 ->> 20[tail]
    print(dcl.length())          # 2

    # Insert head again
    dcl.insert_head(5)
    print(dcl)                   # [head]5 ->> 10 ->> 20[tail]

    # Insert after head (after 5)
    dcl.insert_after(7, dcl.head)
    print(dcl)                   # [head]5 ->> 7 ->> 10 ->> 20[tail]

    # Insert before tail (before 20)
    dcl.insert_before(15, dcl.tail)
    print(dcl)                   # [head]5 ->> 7 ->> 10 ->> 15 ->> 20[tail]

    # Check __contains__ and contains()
    print(10 in dcl)             # True
    print(dcl.contains(25))      # False

    # Access by index
    print(dcl[0])                # 5
    print(dcl[2])                # 10
    # print(dcl[-1])  # No (negative index not supported)

    # Set value at index
    dcl[1] = 8
    print(dcl)                   # [head]5 ->> 8 ->> 10 ->> 15 ->> 20[tail]

    # Traverse using generator
    print(list(dcl.traverse(lambda x: x * 2)))  
    # [10, 16, 20, 30, 40]

    # Reversed iteration
    print(list(reversed(dcl)))   
    # [20, 15, 10, 8, 5]

    # Bidirectional search
    print(dcl.bidirectional_search_value(15, return_node=False))  # 15
    print(dcl.bidirectional_search_value(8, return_node=True))    # Node: 8

    # Search value
    print(dcl.search_value(10, return_node=False))  # 10
    print(dcl.search_value(10, return_node=True))   # Node: 10

    # Search all values
    dcl.insert_tail(10)  
    print([n.data for n in dcl.search_all_values(10, return_node=True)])  # [10, 10]

    # Search index by value
    print(dcl.search_for_index_by_value(10))       # 2 (first occurrence)
    print(dcl.search_for_index_by_value(10, reverse=True)) # 5 (last occurrence, after adding tail 10)

    # Delete head
    print(dcl.delete_head())    # 5
    print(dcl)                  # [head]8 ->> 10 ->> 15 ->> 20 ->> 10[tail]

    # Delete tail
    print(dcl.delete_tail())    # 10
    print(dcl)                  # [head]8 ->> 10 ->> 15 ->> 20[tail]
    dcl.clear()

    # Rotation test with integers
    dcl.clear()
    for val in [1, 2, 3, 4]:
        dcl.insert_tail(val)
    print("Original list:", dcl)

    dcl.single_rotate_left()
    print("After single_rotate_left:", dcl)

    dcl.rotate_left(2)
    print("After rotate_left(2):", dcl)

    dcl.single_rotate_right()
    print("After single_rotate_right:", dcl)

    dcl.rotate_right(3)
    print("After rotate_right(3):", dcl)


if __name__ == "__main__":
    main()
