from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast, Iterator, Generator
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes


"""
Queue ADT Definition:
Formally, the queue abstract data type defines a collection that keeps objects in a sequence.
Where element access & deletion are restricted to the first element in the queue.
Element insertion is restricted to the back of the sequence.

Operations:
enqueue(Q, x): Add element x to the rear of the queue Q
dequeue(Q) -> x: remove & return the first element from the queue Q. An error will be raised if the queue Q is empty
peek(Q) -> x: return the first element (but do not remove it.)

Properties / Constraints:
FIFO principle: First element enqueued is first dequeued.
Non-commutative insertion: Order of enqueues affects order of dequeues.
Empty queue identity: A newly created queue has no elements.
Dequeue on empty queue: Raises error/exception.
"""
T = TypeVar('T')

class QueueADT(ABC, Generic[T]):

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def enqueue(self, value: T):
        """Adds an Element to the end of the Queue"""
        pass

    @abstractmethod
    def dequeue(self) -> T:
        """remove and return the first element of the Queue"""
        pass

    @abstractmethod
    def peek(self) -> T:
        """return (but not remove) the first element of the Queue"""
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


CTYPES_DATATYPES = {
    int: ctypes.c_int,
    float: ctypes.c_double,
    bool: ctypes.c_bool,
    str: ctypes.py_object,  # arbitrary Python object (strings)
    object: ctypes.py_object,  # any Python object
}

NUMPY_DATATYPES = {
    int: numpy.int32,
    float: numpy.float64,
    bool: numpy.bool_,
}


class StaticArrayQueue(QueueADT[T]):
    """A Queue Data Structure based on a Static Array"""
    def __init__(self, capacity: int, datatype: type, datatype_map: dict = CTYPES_DATATYPES) -> None:
        # Type Enforcement & Conversion
        self.datatype = datatype
        self.datatype_map = datatype_map
        self.type = None

        # boundary check for initial capacity:
        if capacity < 2:
            raise ValueError("Queue must be bigger than 1 element.")

        # Queue Core Components
        self.capacity = capacity
        self.data = self._initialize_new_array()   # stores all the elements in an array.
        self.size = 0 # tracks the number of elements in the list.
        self.front = 0 # index of the oldest (first) element

    # ----- Utility -----
    def _initialize_new_array(self):
        """chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)"""
        if self.datatype_map == CTYPES_DATATYPES:
            new_array = self._init_ctypes_array()
        elif self.datatype_map == NUMPY_DATATYPES:
            new_array = self._init_numpy_array()
        else:
            raise ValueError(f"Datatype Map Unknown... Map: {self.datatype_map}")
        return new_array

    def _init_ctypes_array(self):
        """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
        # setting ctypes datatype -- needed for the array.
        if self.datatype not in self.datatype_map:
            self.type = ctypes.py_object
        else:
            self.type = CTYPES_DATATYPES[self.datatype] # maps type of array to ctype
        # creates a class object - an array of specified number of a specified type
        static_array_cls = self.type * self.capacity
        new_array = static_array_cls() # initializes array with preallocated memory block
        return new_array

    def _init_numpy_array(self):
        """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
        if self.datatype not in NUMPY_DATATYPES:
            self.type = object
            new_array = numpy.empty(self.capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            new_array = numpy.empty(self.capacity, dtype=self.type)
        return new_array

    def _underflow_error(self):
        if self.is_empty():
            raise ValueError("Error: Queue is empty.")

    def _overflow_error(self):
        if self.size >= self.capacity:
            raise OverflowError("Error: Queue is full.")

    def __str__(self):
        """a list of strings representing all the elements in the stack. Goes from Top to Bottom"""
        items = ", ".join(str(self.data[(self.front + i) % self.capacity]) for i in range(self.size))
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        return f"Type: {string_datatype}: Capacity: {self.size}/{self.capacity} Queue Items: [Front]->> {items} <<-[Rear]"

    # ----- Canonical ADT Operations -----
    def enqueue(self, value):
        """
        Adds an Element to the rear of the Queue: Utilizes a Circular Buffer 
        Step 1: Check if Queue is Full
        Step 2: Compute rear value = (front + size) % capacity
        Step 3: Assign Value to rear index
        Step 4: Increment Queue Size Tracker
        """
        self._overflow_error()
        # type enforcement:
        if not isinstance(value, self.datatype):
            raise TypeError(f"Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}")
        # compute rear value - where to index the next element.
        rear = (self.front + self.size) % self.capacity
        self.data[rear] = value # assign value
        self.size += 1  # increment size tracker

    def dequeue(self):
        """
        Remove and return the first element of the Queue
        Step 1: Check if Queue is empty
        Step 2: Store current Front element for returning.
        Step 3: Dereference the current Front element.
        Step 4: Increment Front Tracker via Modulo Arithmetic
        Step 5: Decrement Size Tracker (Queue size has decreased)
        Step 6: return deleted value
        """
        self._underflow_error()
        deleted = self.data[self.front] # store deleted value (front)
        # dereference element if its an object. otherwise just unlink
        if self.type is object or self.type is ctypes.py_object: 
            self.data[self.front] = None    
        self.front = (self.front + 1) % self.capacity    # increment front via modulo arithmetic
        self.size -= 1 # decrement size tracker
        return deleted

    def peek(self):
        """
        return (but not remove) the first element of the Queue
        Step 1: Check if Queue is empty
        Step 2: Store Front element
        Step 3: return Front Element
        """
        self._underflow_error()
        result = self.data[self.front]
        return result

    # ----- Meta Collection ADT Operations -----
    def is_empty(self):
        return self.size == 0

    def __len__(self):
        return self.size

    def clear(self):
        """Clear the queue, safely handling object arrays"""
        if self.type is object or self.type is ctypes.py_object:
            for i in range(self.size):
                # uses modulo arithmetic to start from front of queue
                index = (self.front + i) % self.capacity    
                self.data[index] = None

        self.size = 0
        self.front = 0

    def __contains__(self, value):
        """Does the Queue Contain this item? return true or false"""
        for i in range(self.size):
            index = (self.front + i) % self.capacity
            if self.data[index] == value:
                return True
        return False

    def __iter__(self):
        """for use with lists and loops - generator"""
        for i in range(self.size):
            index = (self.front + i) % self.capacity
            yield self.data[index]


# Main --- Client Facing Code ---

def main():
    print("=== Test 1: Integer Queue ===")
    q_int = StaticArrayQueue[int](5, int)
    print("Empty?", q_int.is_empty())  # True
    print("Length:", len(q_int))  # 0
    print(q_int)  # Should show empty queue

    q_int.enqueue(10)
    q_int.enqueue(20)
    q_int.enqueue(30)
    print("After enqueue 10,20,30:", q_int)

    print("Peek:", q_int.peek())  # 10
    print("Contains 20?", 20 in q_int)  # True
    print("Contains 40?", 40 in q_int)  # False

    print("Dequeue:", q_int.dequeue())  # 10
    print("After dequeue 10:", q_int)
    print("Dequeue:", q_int.dequeue())  # 20
    print("After dequeue 20:", q_int)
    print("Peek after dequeue:", q_int.peek())  # 30
    print("Length after dequeue:", len(q_int))  # 1

    q_int.enqueue(40)
    q_int.enqueue(50)
    q_int.enqueue(60)
    print("After circular enqueue 40,50,60:", q_int)

    q_int.clear()
    print("After clear:", q_int)
    print("Length after clear:", len(q_int))  # 0
    print("Empty after clear?", q_int.is_empty())  # True

    # --- Overflow Handling ---
    print("\n=== Test 2: Overflow Handling ===")
    try:
        for i in range(6):
            q_int.enqueue(i)  # 6th enqueue should fail
    except OverflowError as e:
        print("Overflow caught:", e)
    print("Queue after overflow test:", q_int)

    # --- Underflow Handling ---
    print("\n=== Test 3: Underflow Handling ===")
    q_empty = StaticArrayQueue[str](3, str)
    try:
        q_empty.dequeue()  # Should raise ValueError
    except ValueError as e:
        print("Underflow caught:", e)
    print("Queue after underflow test:", q_empty)

    # --- Type Checking ---
    print("\n=== Test 4: Type Checking ===")
    q_str = StaticArrayQueue[str](3, str)
    q_str.enqueue("hello")
    print("Queue after enqueue 'hello':", q_str)
    try:
        q_str.enqueue(123)  # Should raise TypeError
    except TypeError as e:
        print("Type error caught:", e)
    print("Queue after type test:", q_str)

    # --- Boolean Queue ---
    print("\n=== Test 5: Boolean Queue ===")
    q_bool = StaticArrayQueue[bool](3, bool)
    q_bool.enqueue(True)
    q_bool.enqueue(False)
    print("Boolean queue:", q_bool)

    # --- Dynamic Class Test ---
    print("\n=== Test 6: Dynamic Class Type Enforcement ===")
    MyClass = type(
        "MyClass", (), {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__repr__": lambda self: f"MyClass({self.name})",  # <-- this makes str/print show nicely
            }
    )
    OtherClass = type(
        "OtherClass", (), {
            "__init__": lambda self, name: setattr(self, "name", name),
            "__repr__": lambda self: f"MyClass({self.name})",  # <-- this makes str/print show nicely
            }
    )

    q_dyn = StaticArrayQueue[MyClass](5, MyClass)
    a = MyClass("Alice")
    b = MyClass("Bob")
    q_dyn.enqueue(a)
    q_dyn.enqueue(b)
    print("Queue after enqueue Alice and Bob:", q_dyn)

    try:
        invalid = OtherClass("Charlie")
        q_dyn.enqueue(invalid)  # Should raise TypeError
    except TypeError as e:
        print("Type error caught:", e)
    print("Queue after attempting invalid enqueue:", q_dyn)

    first = q_dyn.dequeue()
    print("Dequeued:", first.name, type(first).__name__)
    print("Queue after dequeue:", q_dyn)
    print("Peek:", q_dyn.peek().name, type(q_dyn.peek()).__name__)
    print("Queue currently:", q_dyn)

    print("Contains Bob?", b in q_dyn)
    print("Contains Alice?", a in q_dyn)

    q_dyn.clear()
    print("Queue after clear:", q_dyn)
    print("Length after clear:", len(q_dyn))
    print("Empty after clear?", q_dyn.is_empty())


if __name__ == "__main__":
    main()
