from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Generator, Iterator, Iterable
from abc import ABC, ABCMeta, abstractmethod


"""
**Linked List** is a linear data structure where elements (called nodes) are connected using pointers/references rather than stored in contiguous memory like an array.

Properties:
Ordered: maintains linear sequence of elements.
Dynamic size: grows or shrinks at runtime.
Non-contiguous memory: nodes allocated individually.
Sequential access: access by index requires traversal (O(n)).
"""

T = TypeVar('T')

# Interfaces
class iLinkedList(ABC, Generic[T]):

    @abstractmethod
    def insert_head(self, node_data):
        """O(1) -- we just update the head pointer."""
        pass

    @abstractmethod
    def insert_tail(self, node_data):
        """O(N) -- in a singly linked list (unless you maintain a tail pointer)"""
        pass

    @abstractmethod
    def insert_at(self, index: int, node_data):
        """O(N) -- Inserting elsewhere, because you have to traverse (for loop)."""
        pass

    @abstractmethod
    def delete_at(self, index: int) -> Optional[T]:
        """O(N) -- because you have to traverse the linked list (via .next) """
        pass

    @abstractmethod
    def delete_head(self) -> Optional[T]:
        """O(1) -- we just remove the head."""
        pass

    @abstractmethod
    def search_by_index(self, index: int) -> Optional[T]:
        pass

    @abstractmethod
    def search_for_index(self, node_data) -> int | None:
        pass

    @abstractmethod
    def contains(self, node_data) -> bool:
        pass

    @abstractmethod
    def traverse(self, function: Callable) -> Generator["Node" | T, None, None]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def length(self) -> int:
        pass


class iNode(ABC):
    def __init__(self, data) -> None:
        self.data = data
        self.next = None

# Concrete Classes
class Node(iNode):
    """Node Element in Linked List"""
    def __init__(self, data) -> None:
        self.data = data
        self._next: Optional[iNode] = None    # initialized as none, stores a reference to the next node in the linked list

    @property
    def next(self): 
        return self._next
    @next.setter
    def next(self, value):
        self._next = value

    def __repr__(self) -> str:
        return f"Node: {self.data}"


class LinkedList(iLinkedList[T]):
    """Implements a Singly Linked List"""
    def __init__(self) -> None:
        self.head: Optional[iNode] = None # first node in the linked list
        self._counter: int = 0  # tracks the number of nodes in the linked list





    def __iter__(self):
        """Allows iteration over the Nodes in the linked list. (via for loops etc)"""        
        current_node = self.head
        while current_node:
            yield current_node.data
            current_node = current_node.next

    def clear(self):
        while not self.is_empty():
            self.delete_head()

    def insert_head(self, node_data):
        """This method inserts a new node at the start(head) of the linked list."""
        new_node = Node(node_data)  # create new node object with user data
        new_node._next = self.head   # next -> points to current head (which becomes the next item in the list)
        self.head = new_node    # the new node becomes the new head
        self._counter += 1  # update the linked list elements tracker

    def insert_tail(self, node_data):
        """Inserts a new node at the end (tail) of the linked list."""
        new_node = Node(node_data)
        # if the list is empty - insert at head (which is also the tail)
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head    # start from the head
            # travel through the linked list unti we get to the end (current_node.next = None)
            while current_node.next:
                current_node = current_node.next
            current_node.next = new_node    # at the end - assign new node and data
        self._counter += 1  # update the linked list size tracker

    def insert_at(self, index, node_data):
        """Inserts a new node at a specified index position. Updates the rest of the list accordingly"""
        # Validate Boundaries
        if index < 0 or index > self._counter:
            raise IndexError("Index Out of Bounds")

        # if index is the head - use O(1) method for performance
        if index == 0:
            self.insert_head(node_data)
            return

        # STEP 1: initialize New Node & Initialize Head as the current node
        new_node, current_node= Node(node_data), self.head
        # STEP 2: identify the current node (stop at prior node)
        for i in range(index - 1):
            current_node = current_node.next    # assign current node via next
        # STEP 3: Link the new node to the rest of the list (elements after new_node)
        new_node._next = current_node.next
        # STEP 4: add newnode into the list after the current node (elements before new_node)
        current_node.next = new_node
        # STEP 5: update size tracker
        self._counter += 1  

    def delete_head(self):
        """Deletes the Node at the head - and returns the data for inspection"""
        # existence check
        if self.head is None:
            raise IndexError("Cant Delete Head from an Empty List")
        removed_head = self.head
        self.head = self.head.next
        self._counter -= 1
        return removed_head.data

    def delete_at(self, index):
        """Deletes a node from a specified index"""
        # existence check
        if self.head is None:
            raise IndexError("Cant Delete Head from an Empty List")

        # Validate Boundaries
        if index < 0 or index >= self._counter:
            raise IndexError("Index Out of Bounds")

        # if index is the head - use O(1) method for performance
        if index == 0:
            self.delete_head()
            return

        # STEP 1: initialize node traversal
        current_node = self.head
        # STEP 2: travel to 1 before index
        for _ in range(index -1):
            current_node = current_node.next
        # STEP 3: label the node target(index) for removal(dereferencing)
        removed = current_node.next
        # STEP 4: reroute linked list to bypass removed node
        current_node.next = removed.next
        # STEP 5: update the size tracker
        self._counter -= 1
        # STEP 6: return the data from the removed node
        return removed.data

    def search_by_index(self, index):
        """Finds the Node at a specific index"""
        if index < 0 or index >= self._counter:
            raise IndexError("Index Out of Bounds")

        current_node = self.head
        # node at current index position
        for i in range(index):
            current_node = current_node.next # pyright: ignore[reportOptionalMemberAccess]
        return current_node.data # pyright: ignore[reportOptionalMemberAccess]

    def search_for_index(self, node_data) -> int | None:
        """Return the index position of the first node that contains the value."""
        # initialize
        current_node = self.head
        index = 0
        while current_node:
            if current_node.data == node_data:
                return index
            current_node = current_node.next
            index += 1
        return None # if data not found return None

    def contains(self, node_data) -> bool:
        """return True or False if a node contains the specified data."""
        current_node = self.head
        while current_node:
            if current_node.data == node_data:
                return True
            current_node = current_node.next
        return False

    def traverse(self, function):
        """executes a function on every node in the linked list"""
        current_node = self.head
        while current_node:
            try: 
                yield function(current_node.data)
            except Exception as error:
                print(f"There was an error while trying to apply function to the node: {current_node.data}: {error}")
            finally:
                current_node = current_node.next

    def is_empty(self) -> bool:
        return self.head is None

    def length(self) -> int:
        return self._counter

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













# Main -- Client Facing Code ---
def main():
    # ====== TESTING LINKED LIST ======

    linklist = LinkedList[Any]()

    print("1. Initial empty list")
    print(linklist)
    print("Is empty?", linklist.is_empty())
    print("Length:", linklist.length())
    print()

    # 2. Insert mixed types at head and tail
    linklist.insert_head(100)            # int
    linklist.insert_tail("hello")        # str
    linklist.insert_head([1, 2, 3])      # list
    linklist.insert_tail({"key": "val"}) # dict
    linklist.insert_head(3.14)           # float
    linklist.insert_tail("hello")        # duplicate str

    print("2. After inserting mixed types including duplicates:")
    print(linklist)
    print("Length:", linklist.length())
    print()

    # 3. Insert at various indices
    linklist.insert_at(0, "start")   # insert at head
    linklist.insert_at(3, "middle")  # insert in middle
    linklist.insert_at(linklist.length(), "end") # insert at tail

    print("3. After insert_at operations:")
    print(linklist)
    print("Length:", linklist.length())
    print()

    # 4. Search by index
    print("4. search_by_index examples:")
    print("Index 0:", linklist.search_by_index(0))
    print("Index 3:", linklist.search_by_index(3))
    print("Last index:", linklist.search_by_index(linklist.length() - 1))
    print()

    # 5. Search for value (including duplicates)
    print("5. search_for_index examples:")
    print("First 'hello':", linklist.search_for_index("hello"))  # should return first occurrence
    print("First 100:", linklist.search_for_index(100))
    print("Nonexistent:", linklist.search_for_index("absent"))
    print()

    # 6. Membership tests
    print("6. contains examples:")
    print("'middle' in list?", linklist.contains("middle"))
    print("'absent' in list?", linklist.contains("absent"))
    print()

    # 7. Delete head, tail, and middle
    print("7. Deletion tests:")
    print("Delete head:", linklist.delete_head())
    print("Delete index 3:", linklist.delete_at(3))
    print("Delete last index:", linklist.delete_at(linklist.length() - 1))
    print("List after deletions:")
    print(linklist)
    print("Length:", linklist.length())
    print()

    # 8. Traversal with a function
    print("8. Traverse with function (converts strings only):")
    def uppercase_strings(item):
        if isinstance(item, str):
            return item.upper()
        else:
            return item

    # Using traverse() generator
    for result in linklist.traverse(uppercase_strings):
        print(result)

    # 9. __iter__ usage
    print("9. iterate via __iter__ and for loop:")
    for item in linklist:
        print(item)

    # 10. Clear the list
    print("10. Clear list:")
    linklist.clear()
    print(linklist)
    print("Is empty?", linklist.is_empty())
    print("Length:", linklist.length())


if __name__ == "__main__":
    main()
