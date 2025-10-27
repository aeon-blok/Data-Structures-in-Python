from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast
from abc import ABC, ABCMeta, abstractmethod

# array adt

"""
**Dynamic Array**: collection of elements of type E in linear order
A contiguous block of memory that resizes automatically when it runs out of space.

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


# Generic Type
T = TypeVar("T")


# interface
class iDynamicArray(ABC, Generic[T]):
    """"""

    @abstractmethod
    def create(self, size) -> None:
        """O(n)"""
        pass

    @abstractmethod
    def get(self, index) -> T:
        """O(1)"""
        pass

    @abstractmethod
    def set(self, index, value: T):
        """O(1)"""
        pass

    @abstractmethod
    def size(self) -> int:
        """O(1)"""
        pass

    @abstractmethod
    def _resize(self, new_capacity: int) -> None:
        """O(1)"""
        pass

    @abstractmethod
    def insert(self, index, value: T):
        """O(n)"""
        pass

    @abstractmethod
    def delete(self, index: int) -> Any:
        """O(n)"""
        pass

    @abstractmethod
    def traverse(self, function: Callable[[T], Any]):
        """O(n)"""
        pass

    @abstractmethod
    def search(self, value: T) -> Any:
        """O(n)"""
        pass


class DynamicArray(iDynamicArray[T]):
    """Dynamic Array — automatically resizes as elements are added."""

    MIN_CAPACITY: int =2

    def __init__(self, capacity: int = MIN_CAPACITY) -> None:
        self._capacity = capacity  # fixed size
        self._state: List[Optional[T]] = [None] * self._capacity  # array is empty at creation
        self._counter: int = 0  # counts the number of elements currently stored in the array, needed for the size method

    # private methods
    def _resize(self, new_capacity: int) -> None:
        """O(n) — Allocates new space and copies elements."""
        temp_state: list[Optional[T]]= [None] * new_capacity
        for i in range(self._counter):
            temp_state[i] = self._state[i]  # copy all the elements from previous sized array.
        self._state = temp_state
        self._capacity = new_capacity

    # public methods
    def create(self, size) -> None:
        """Creates an Array of a specific size, resets if the array is already created...."""
        self._state = [None] * size
        self._capacity = size
        self._counter = 0

    def get(self, index: int) -> T:
        """O(1) returns the element at the specified index"""
        if 0 <= index < self._counter:  # boundary check using chained comparison syntax
            return cast(T, self._state[index])
        raise IndexError("Index out of bounds")

    def set(self, index: int, value: T):
        """O(1) Updates Element at specified index"""

        if 0 <= index < self._counter:
            self._state[index] = value
        else:
            raise IndexError("Index out of bounds")

    def size(self) -> int:
        """returns the number of elements"""
        return self._counter

    def insert(self, index: int, value: T):
        """O(n) inserts a value at the specified index. Shifts elements right."""

        if not (0 <= index <= self._counter):
            raise IndexError("Index out of bounds")

        if self._counter == self._capacity:
            self._resize(self._capacity * 2)    # resizes array *2

        # move all array elements right.
        for i in range(self._counter, index, -1):  # start from the end and go backwards through array. (stop at index element)
            self._state[i] = self._state[i - 1]  # copies element from left neighbour (e.g. elem_4 = elem_3) - shifts every element right
        self._state[index] = value  # insert value into the array at index (atm contains duplicate)
        self._counter += 1  # increment array counter

    def append(self, value: T):
        """O(1) -- inserts a value at the end of the array"""
        # resize if array is full
        if self._counter == self._capacity:
            self._resize(self._capacity * 2)
        # store value at the end of currently stored items
        self._state[self._counter] = value  
        self._counter += 1  # increment counter

    def delete(self, index: int):
        """O(n) removes a value at the specified index, Shifts elements left"""

        if not 0 <= index < self._counter:
            raise IndexError("Index out of bounds")

        delete_value = self._state[index]  # returned at end for user info

        # shift elements left -- Starts from the deleted index (Goes Backwards)
        # Stops before the index (the item we will remove)
        for i in range(index, self._counter - 1):  
            # copies element from right neighbour (elem4 = elem5) - shifts every element left
            self._state[i] = self._state[i + 1] 
        # removes item from the end of the stored items
        self._state[self._counter - 1] = None  
        self._counter -= 1  # decrement array counter

        # shrink array if array is not empty & current capacity <= 25% of array size
        if 0 < self._counter <= self._capacity // 4:  
            # halves the size of the array with min value of 2
            self._resize(max(DynamicArray.MIN_CAPACITY, self._capacity // 2))  

        # This is useful if the caller wants to use or log the deleted value.
        return cast(T, delete_value)  

    def traverse(self, function: Callable[[T], Any]):
        """Applies a function to each element"""
        if not callable(function):
            raise TypeError("Invalid Function Entered!")
        for i in range(self._counter):
            function(cast(T, self._state[i]))  # cast fixes type hint

    def search(self, value: T) -> Any:
        """O(n) returns index of value if present, else -1"""
        for i in range(self._counter):
            if self._state[i] == value:
                return i
        # loop completes without finding value -- signals value is not present in array.
        return None

    # Utility Methods
    def capacity(self):
        """returns the current size of the array"""
        return self._capacity

    def __str__(self) -> str:
        """Info string for Array"""
        elements = [self._state[i] for i in range(self._counter)]
        return (
            f"Array Size: {self._counter}/{self._capacity}: With Elements: {elements}"
        )

    def __repr__(self) -> str:
        """Shows the array with all the empty positions also"""
        return f"Array: {self._state}"


# Main -- Client Facing Code
if __name__ == "__main__":

    # initialize array
    array = DynamicArray[int]()
    array.create(3)

    # Insert elements
    array.insert(0, 10)
    array.insert(1, 20)
    array.insert(2, 30)
    array.insert(3, 40)
    array.insert(4, 50)
    array.insert(5, 60)

    # test infostrings
    print(array)  # __str__
    print(repr(array))  # __repr__

    # Test get
    print("\nGet element at index 2:")
    print(array.get(2))  # 30

    # Test set
    print("\nSet element at index 2 to 35")
    array.set(2, 35)
    print(array.get(2))  # 35

    # Test size
    print("\nCurrent size:")
    print(array.size())  # 5

    # Test search
    print("\nSearch for value 40:")
    print(array.search(40))  # 3
    print("Search for value 100 (not in array):")
    print(array.search(100))  # -1

    # Test traverse
    print("\nTraverse (multiply each element by 2 and print):")
    array.traverse(lambda x: print(x * 2))

    # Test delete
    print("\nDelete element at index 1:")
    print(repr(array))
    deleted = array.delete(1)
    print(f"Deleted value: {deleted}")
    print(repr(array))

    # Test insert at middle
    print("\nInsert 25 at index 1:")
    array.insert(1, 25)
    print(array)
