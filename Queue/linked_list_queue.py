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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


# Custom Types
T = TypeVar("T")


# Interfaces

class iNode(ABC, Generic[T]):
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class QueueADT(ABC, Generic[T]):

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def enqueue(self, value: T):
        """O(1) -- Adds an Element to the end of the Queue"""
        pass

    @abstractmethod
    def dequeue(self) -> T:
        """O(1) -- remove and return the first element of the Queue"""
        pass

    @abstractmethod
    def peek(self) -> T:
        """O(1) -- return (but not remove) the first element of the Queue"""
        pass

    # ----- Meta Collection ADT Operations -----
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def __contains__(self, value: T) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        pass


# Concrete Implementations

class Node(iNode[T]):
    def __init__(self, data: T) -> None:
        self.data = data
        self.next: Optional[Node] = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.data}"


class LinkedListQueue(QueueADT[T]):
    """Linked List Queue: utilizes a singly linked list with head & tail pointers."""
    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size: int = 0

    # ----- Utility -----
    def _underflow_error(self):
        if not self.head:
            raise IndexError("Error: Queue is empty")

    def __str__(self) -> str:
        """Displays all the content of the linked list queue as a string."""

        seperator = " ->> "

        if self.head is None:
            return f"List is Empty"

        def _simple_traversal():
            """traverses the nodes and returns a string via generator"""
            current_node = self.head
            while current_node:
                yield str(current_node.data)
                current_node = current_node.next

        return f"[head]{seperator.join(_simple_traversal())}[tail]"
    
    def __repr__(self) -> str:
        """returns the classname and the size of the queue"""
        return f"{self.__class__.__name__}(size={self.size})"

    # ----- Canonical ADT Operations -----
    def enqueue(self, value):
        """
        Adds an Element to the end of the Queue
        Step 1: initialize Node
        Step 2: if Tail exists - point to the new node
        Step 3: New Node becomes the Tail
        Step 4: If list is empty - new node becomes the head
        Step 5: update queue size value.
        """
        node = Node(value)  # initialize Node Element

        # if a tail exists - points it to the new node.
        if self.tail:
            self.tail.next = node
        # New node becomes the tail. (last node)
        self.tail = node

        # if the queue is empty - Node becomes the head (first element - aka the front)
        if not self.head:
            self.head = node

        self.size += 1  # increment queue size

    def dequeue(self):
        """
        remove and return the first element of the Queue
        Step 1: Check if queue is empty
        step 2: store deleted value (the head) to return later
        Step 3: Move Head to the next node in the chain (deleting the original head)
        Step 4: Dereference head and tail if list is now empty
        Step 5: Update queue size tracker
        Step 6: return deleted value (the original head.)
        """

        self._underflow_error()

        deleted = self.head.data
        self.head = self.head.next
        # Dereferencing: if the queue becomes empty after removing a link. ensure both head and tail are none
        if self.head is None:
            self.tail = None
        self.size -= 1
        return deleted

    def peek(self):
        """return (but not remove) the first element of the Queue"""
        self._underflow_error()
        return self.head.data

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        return self.head is None

    def __len__(self):
        return self.size

    def clear(self):
        """Clearing a linked list queue"""        
        # Traverse the list
        current_node = self.head
        while current_node:
            next_node = current_node.next   # 1 after current node
            # dereference current node
            current_node.next = None
            current_node = next_node
        # derefence the Head and Tail
        self.head = None
        self.tail = None
        self.size = 0   # update tracker.

    def __contains__(self, value):
        """Check if the queue contains a value - returns true or false"""
        if self.head is None:
            return False
        current_node = self.head
        while current_node:
            if current_node.data == value:
                return True
            current_node = current_node.next
        return False

    def __iter__(self):
        """allows for iterations - loops, lists etc..."""
        current_node = self.head
        while current_node:
            yield current_node.data
            current_node = current_node.next


# Main --- Client Facing Code ---
def main():
    print("\n=== LINKED LIST QUEUE TEST SUITE WITH ERROR CHECKS ===")

    # --- Empty Queue Error Checks ---
    print("\n--- Empty Queue Error Checks ---")
    q_empty = LinkedListQueue[int]()
    try:
        q_empty.dequeue()
    except IndexError as e:
        print("Dequeue on empty queue:", e)
    try:
        q_empty.peek()
    except IndexError as e:
        print("Peek on empty queue:", e)

    # --- Integer Queue ---
    print("\n--- Testing Integer Queue with 5 items ---")
    q_int = LinkedListQueue[int]()
    for i in range(1, 6):
        q_int.enqueue(i)
    print("Queue:", q_int)
    print("Peek:", q_int.peek())
    print("Contains 3?", 3 in q_int)
    print("Contains 10? (should be False)", 10 in q_int)
    print("Length:", len(q_int))
    print("Dequeueing all items:")
    while not q_int.is_empty():
        print(q_int.dequeue(), end=" ")
    print()
    try:
        q_int.dequeue()
    except IndexError as e:
        print("Dequeue on empty after clearing:", e)

    # --- Dynamically Generated Classes ---
    MyClassA = type(
        "MyClassA",
        (object,),
        {
            "__init__": lambda self, id, name: setattr(self, "_data", (id, name)),
            "__repr__": lambda self: f"MyClassA(id={self._data[0]}, name='{self._data[1]}')",
            "__eq__": lambda self, other: isinstance(other, type(self))
            and self._data == other._data,
        },
    )
    MyClassB = type(
        "MyClassB",
        (object,),
        {
            "__init__": lambda self, code, value: setattr(self, "_data", (code, value)),
            "__repr__": lambda self: f"MyClassB(code={self._data[0]}, value='{self._data[1]}')",
            "__eq__": lambda self, other: isinstance(other, type(self))
            and self._data == other._data,
        },
    )

    # --- Custom Object Queue with Error Checks ---
    print("\n--- Testing Custom Object Queue (Type Enforcement) ---")
    obj_a1 = MyClassA(1, "Alice")
    obj_a2 = MyClassA(2, "Bob")
    obj_b1 = MyClassB(99, "X")

    q_obj = LinkedListQueue[MyClassA]()
    q_obj.enqueue(obj_a1)
    q_obj.enqueue(obj_a2)
    print("Queue:", q_obj)
    print("Contains obj_a2?", obj_a2 in q_obj)
    print("Contains obj_b1? (should be False)", obj_b1 in q_obj)
    print("Peek:", q_obj.peek())
    print("Dequeueing all items:")
    while not q_obj.is_empty():
        print(q_obj.dequeue(), end=" ")
    print()
    try:
        q_obj.peek()
    except IndexError as e:
        print("Peek on empty object queue:", e)

    # --- Mixed Types Queue with Any ---
    print("\n--- Testing Mixed Types Queue ---")
    q_any = LinkedListQueue[Any]()
    q_any.enqueue(1)
    q_any.enqueue("a")
    q_any.enqueue(obj_a1)
    q_any.enqueue(obj_b1)
    print("Queue:", q_any)
    print("Iterating:")
    for item in q_any:
        print(item, "Type:", type(item))

    print("\n=== ERROR CHECKS COMPLETE ===")


if __name__ == "__main__":
    main()
