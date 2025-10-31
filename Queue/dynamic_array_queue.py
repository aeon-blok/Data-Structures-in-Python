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
T = TypeVar("T")


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


class DynamicArrayQueue(QueueADT[T]):
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
        self.min_capacity = max(4, capacity)
        self.data = self._initialize_new_array(self.capacity)  # stores all the elements in an array.
        self.size = 0  # tracks the number of elements in the list.
        self.front = 0  # index of the oldest (first) element

    # ----- Utility -----
    def _initialize_new_array(self, capacity):
        """chooses between using CTYPE or NUMPY style array - CTYPES are more flexible (can have object arrays...)"""
        if self.datatype_map == CTYPES_DATATYPES:
            new_array = self._init_ctypes_array(capacity)
        elif self.datatype_map == NUMPY_DATATYPES:
            new_array = self._init_numpy_array(capacity)
        else:
            raise ValueError(f"Datatype Map Unknown... Map: {self.datatype_map}")
        return new_array

    def _init_ctypes_array(self, capacity):
        """Creates a CTYPES array - much faster than standard python list. but is fixed in size and restricted in datatypes it can use..."""
        # setting ctypes datatype -- needed for the array.
        if self.datatype not in self.datatype_map:
            self.type = ctypes.py_object
        else:
            self.type = CTYPES_DATATYPES[self.datatype]  # maps type of array to ctype
        # creates a class object - an array of specified number of a specified type
        static_array_cls = self.type * capacity
        new_array = static_array_cls() # initializes array with preallocated memory block
        return new_array

    def _init_numpy_array(self, capacity):
        """Creates a Numpy array - much faster than standard python list, but fixed in size, and much more restricted in the datatypes it can use..."""
        if self.datatype not in NUMPY_DATATYPES:
            self.type = object
            new_array = numpy.empty(capacity, dtype=self.type)
        else:
            self.type = NUMPY_DATATYPES[self.datatype]
            new_array = numpy.empty(capacity, dtype=self.type)
        return new_array

    def _underflow_error(self):
        if self.is_empty():
            raise ValueError("Error: Queue is empty.")

    def _overflow_error(self):
        if self.size >= self.capacity:
            raise OverflowError("Error: Queue is full.")

    def __str__(self):
        """a list of strings representing all the elements in the stack. Goes from Top to Bottom"""
        items = ", ".join(
            str(self.data[(self.front + i) % self.capacity]) for i in range(self.size)
        )
        string_datatype = getattr(self.datatype, "__name__", str(self.datatype))
        return f"Type: {string_datatype}: Capacity: {self.size}/{self.capacity} Queue Items: [Front]->> {items} <<-[Rear]"

    def _grow_array(self):
        """
        Step 1: Store existing array data and capacity.
        Step 2: Initialize new array with * 2 capacity
        Step 3: Copy old items to new array, but rearranging so that front is at index 0. using modulo arithmetic. 
        the new array will start indexing from 0 and will copy from the old array front value and loop around.
        Step 4: Reset Front value to 0 - at the start of the new array (where we placed it)
        Step 5: Update the capacity to reflect the new extended capacity.
        Step 6: return the array for use in the program.
        """
        # store exsisting list.
        old_data = self.data
        old_capacity = self.capacity
        # initialize a new array - with double size.
        new_capacity = self.capacity * 2
        new_array = self._initialize_new_array(new_capacity)
        # copy elements from old array to new array -- specific order - from front to end.
        for i in range(self.size):
            index = (self.front + i) % old_capacity
            new_array[i] = old_data[index]

        self.front = 0  # reset front to 0 - start of the new array.
        self.capacity = new_capacity  # update capacity to reflect new size.
        self.data = new_array
        return self.data

    def _shrink_array(self):
        """shrinks array size (by 1/2)"""
        # store exsisting list.
        old_data = self.data
        old_capacity = self.capacity
        # initialize a new array - with double size.
        new_capacity = max(self.min_capacity, self.capacity // 2)
        new_array = self._initialize_new_array(new_capacity)
        # copy elements from old array to new array -- specific order - from front to end.
        for i in range(self.size):
            index = (self.front + i) % old_capacity
            new_array[i] = old_data[index]

        self.front = 0  # reset front to 0 - start of the new array.
        self.capacity = new_capacity  # update capacity to reflect new size.
        self.data = new_array
        return self.data

    # ----- Canonical ADT Operations -----
    def enqueue(self, value):
        """Adds an Element to the rear of the Queue: Utilizes a Circular Buffer"""

        # type enforcement:
        if not isinstance(value, self.datatype):
            raise TypeError(
                f"Invalid Type: Expected: {self.datatype.__name__} Got: {type(value)}"
            )
        elif not issubclass(type(value), self.datatype):
            raise TypeError(f"Is Not a valid subclass of datatype: Expected: {self.datatype.__name__} Got: {type(value)}")

        # dynamically grow array based on size.
        if self.size == self.capacity:
            self.data = self._grow_array()

        # compute rear value - where to index the next element.
        rear = (self.front + self.size) % self.capacity
        self.data[rear] = value  # assign value
        self.size += 1  # increment size tracker

    def dequeue(self):
        """
        Remove and return the first element of the Queue
        Step 1: Check if Queue is empty
        Step 2: Store current Front element for returning.
        Step 3: Dereference the current Front element.
        Step 4: Increment Front Tracker via Modulo Arithmetic
        Step 5: Decrement Size Tracker (Queue size has decreased)
        Step 6: Dynamically Shrink Array if less than 25% capacity
        Step 7: return deleted value
        """

        self._underflow_error()

        deleted = self.data[self.front]  # store deleted value (front)
        # dereference element if its an object. otherwise just unlink
        if self.type is object or self.type is ctypes.py_object:
            self.data[self.front] = None
        # this will loop around the total capacity like a clock giving the correct index as a remainder
        self.front = (self.front + 1) % self.capacity  
        self.size -= 1  # decrement size tracker

        # dynamically shrinks array size if elements are less than 25% of the total capacity.
        if self.size == self.capacity // 4 and self.capacity > self.min_capacity:
            self.data = self._shrink_array()

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
        """
        Clear the queue, safely handling object arrays
        Step 1: Type enforcement & coercion:
        Step 2: reinitialize array and reset capacity back to default
        Step 3: reset size and front values
        """
        if self.type is object or self.type is ctypes.py_object:
            for i in range(self.size):
                # uses modulo arithmetic to start from front of queue
                index = (self.front + i) % self.capacity
                self.data[index] = None

        # reset array and capacity
        if self.capacity != self.min_capacity:
            self.data = self._initialize_new_array(self.min_capacity)
            self.capacity = self.min_capacity

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
    print("\n=== Testing INT Queue ===")
    q_int = DynamicArrayQueue(3, int)
    q_int.enqueue(1)
    q_int.enqueue(2)
    q_int.enqueue(3)
    print(q_int)
    print("Peek:", q_int.peek())
    print("Contains 2?", 2 in q_int)
    print("Length:", len(q_int))
    print("Iterating:", list(q_int))
    q_int.dequeue()
    print("After dequeue:", q_int)
    print("Is empty?", q_int.is_empty())
    q_int.clear()
    print("After clear:", q_int)
    print("Is empty?", q_int.is_empty())

    print("\n=== Testing STRING Queue ===")
    q_str = DynamicArrayQueue(2, str)
    q_str.enqueue("a")
    q_str.enqueue("b")
    print(q_str)
    print("Peek:", q_str.peek())
    try:
        q_str.enqueue(10)  # type enforcement check
    except TypeError as e:
        print("Type Enforcement:", e)
    q_str.dequeue()
    print("After dequeue:", q_str)
    q_str.clear()
    print("After clear:", q_str)

    print("\n=== Testing CUSTOM OBJECT Queue ===")
    MyClass = type(
        "MyClass",
        (),
        {
            "__init__": lambda self, id: setattr(self, "id", id),
            "__repr__": lambda self: f"MyClass({self.id})",
        },
    )
    q_obj = DynamicArrayQueue(2, MyClass)
    obj1 = MyClass(1)
    obj2 = MyClass(2)
    q_obj.enqueue(obj1)
    q_obj.enqueue(obj2)
    print(q_obj)
    print("Peek:", q_obj.peek())
    print("Contains obj1?", obj1 in q_obj)
    print("Iterating:", list(q_obj))
    q_obj.dequeue()
    print("After dequeue:", q_obj)
    q_obj.clear()
    print("After clear:", q_obj)

    print("\n=== Testing LIST Queue ===")
    q_list = DynamicArrayQueue(2, list)
    q_list.enqueue([1, 2])
    q_list.enqueue([3, 4])
    print(q_list)
    print("Peek:", q_list.peek())
    print("Contains [1,2]?", [1, 2] in q_list)
    q_list.dequeue()
    print("After dequeue:", q_list)
    q_list.clear()
    print("After clear:", q_list)

    print("\n=== Testing DICT Queue ===")
    q_dict = DynamicArrayQueue(2, dict)
    q_dict.enqueue({"x": 1})
    q_dict.enqueue({"y": 2})
    print(q_dict)
    print("Peek:", q_dict.peek())
    print("Contains {'x':1}?", {"x": 1} in q_dict)
    q_dict.dequeue()
    print("After dequeue:", q_dict)
    q_dict.clear()
    print("After clear:", q_dict)

    print("\n=== STRESS TEST: GROW & SHRINK ===")
    capacity = 4
    q = DynamicArrayQueue(capacity, int)

    # --- Grow test ---
    print("\n-- Enqueueing to grow --")
    for i in range(20):  # enqueue beyond initial capacity
        q.enqueue(i)
        if i % 5 == 0:  # check periodically
            print(
                f"After enqueue {i}: Size={len(q)}, Capacity={q.capacity}, Front={q.front}"
            )

    print("Queue after growth:", q)
    print("Peek:", q.peek())
    print("Iterating:", list(q))

    # --- Shrink test ---
    print("\n-- Dequeueing to shrink --")
    while not q.is_empty():
        removed = q.dequeue()
        if len(q) % 5 == 0 or q.size <= 5:  # check periodically
            print(
                f"After dequeue {removed}: Size={len(q)}, Capacity={q.capacity}, Front={q.front}"
            )

    print("Queue after shrinking:", q)
    print("Is empty?", q.is_empty())


    # --- Type enforcement check during growth ---
    print("\n-- Type enforcement check --")
    q_str = DynamicArrayQueue(2, str)
    q_str.enqueue("hello")
    try:
        q_str.enqueue(123)  # invalid type
    except TypeError as e:
        print("Caught TypeError as expected:", e)

if __name__ == "__main__":
    main()
