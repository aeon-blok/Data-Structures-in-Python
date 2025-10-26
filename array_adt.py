from typing import Generic, TypeVar, List, Dict, Optional, Callable, Any, cast
from abc import ABC, ABCMeta, abstractmethod

# array adt
"""
**Array**: collection of elements of type E in linear order

Properties / Constraints:
- Elements Stored in linear order
- Random Access via Index allowed
- Size can be fixed or dynamic
- All Elements must be the same type
- Elements stored in Contiguous Memory - In Python: you get contiguous references, not necessarily contiguous objects.
"""


# Generic Type
T = TypeVar('T')

# interface
class iArray(ABC, Generic[T]):
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
    def insert(self, index, value: T):
        """O(n)"""
        pass

    @abstractmethod
    def delete(self, index):
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


class FixedArray(iArray[T]):
    """Fixed Size Collection of elements of type E in linear order"""
    def __init__(self, size: int = 0) -> None:
        self._state: List[Optional[T]] = [None] * size # array is empty at creation
        self._size = size   # fixed size
        self._counter: int = 0   # counts the number of elements currently stored in the array, needed for the size method

    def __str__(self) -> str:
        """Info string for Array"""
        elements = [self._state[i] for i in range(self._counter)]
        return f"Array Size: {self._counter}: With Elements: {elements}"

    def __repr__(self) -> str:
        """Shows the array with all the empty positions also"""
        return f"Array: {self._state}"

    # public methods
    def create(self, size) -> None:
        """Creates an Array of a specific size, resets if the array is already created...."""
        self._state = [None] * size
        self._size = size
        self._counter = 0

    def get(self, index) -> T:
        """O(1) returns the element at the specified index"""
        if 0 <= index < self._counter:  # boundary check using chained comparison syntax
            return self._state[index]
        raise IndexError("Index out of bounds")

    def set(self, index, value: T):
        """O(1) Updates Element at specified index"""

        if 0 <= index < self._counter:
            self._state[index] = value
        else:
            raise IndexError("Index out of bounds")

    def size(self) -> int:
        """returns the number of elements"""
        return self._counter

    def insert(self, index, value: T):
        """O(n) inserts a value at the specified index. Shifts elements right."""

        if self._counter >= self._size:
            raise OverflowError("Array is Full Sir....")

        if not (0 <= index <= self._counter):
            raise IndexError("Index out of bounds")

        # move all array elements right.
        for i in range(self._counter, index, -1):   # start from the end
            self._state[i] = self._state[i-1]   # move element right
        self._state[index] = value  # insert value into the array at index (now there is a space)

        self._counter += 1    # increment array counter

    def delete(self, index):
        """O(n) removes a value at the specified index, Shifts elements left"""

        if not 0 <= index < self._counter:
            raise IndexError("Index out of bounds")

        delete_value = self._state[index]   # returned at end for user info

        # shift elements left -- Starts from the deleted index
        for i in range(index, self._counter - 1):  # Stops before the index (the item we will remove)
            self._state[i] = self._state[i + 1]  # Moves each element one position left
        self._state[self._counter - 1] = None   # removes item from the array

        self._counter -= 1  # decrement array counter

        return delete_value  # This is useful if the caller wants to use or log the deleted value.

    def traverse(self, function: Callable[[T], Any]):
        """Applies a function to each element"""
        if not callable(function):
            raise TypeError("Invalid Function Entered!")
        for i in range (self._counter):
            function(cast(T, self._state[i]))   # cast fixes type hint

    def search(self, value: T) -> Any:
        """O(n) returns index of value if present, else -1"""
        for i in range(self._counter):
            if self._state[i] == value:
                return i
        # loop completes without finding value -- signals value is not present in array.
        return -1 

    # TODO: Add Append - inserts an item at the end of the array.
    # TODO: Why return -1 for null search
    # TODO: Understand Insertion and Deletion logic better...


# Main -- Client Facing Code
if __name__ == "__main__":

    # initialize array
    array = FixedArray[int]()
    array.create(20)

    # Insert elements
    array.insert(0, 10)
    array.insert(1, 20)
    array.insert(2, 30)
    array.insert(3, 40)
    array.insert(4, 50)

    # test infostrings
    print(array)    # __str__
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
