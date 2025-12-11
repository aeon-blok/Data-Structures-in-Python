# region standard imports

from typing import (
    Generic,
    TypeVar,
    Dict,
    Optional,
    Callable,
    Any,
    cast,
    Iterator,
    Generator,
    Iterable,
    TYPE_CHECKING,
    NewType,
    List,
    Literal,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
import math

from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, Index
from user_defined_types.hashtable_types import NormalizedFloat

from utils.validation_utils import DsValidation
from utils.exceptions import *

if TYPE_CHECKING:
    from adts.collection_adt import CollectionADT
    from adts.sequence_adt import SequenceADT

from ds.primitives.arrays.dynamic_array import VectorArray

# endregion

# todo add the ability to use min heap also - lots of logic adjustments....

class HeapSort:
    """
    Heapsort Algorithm Implementation: (unstable, inplace)
    converts an array into a heap
    repeatedly removes the largest or smallest node (the root) and places it at the end of the array. (this is the sorted subsection of the array)
    restores the heap property & repeats the loop until all elements are sorted
    O(N Log N) Time Complexity.
    Useful in memory constrained environments
    """
    def __init__(self, technique: Literal['bottom_up', 'floyd', 'out_of_place'] = 'floyd') -> None:
        self._technique = technique

    # --------------- Heapify Operations ---------------
    def _bottom_up_heapify(self, input_array, i, heap_size):
        """
        Used with Bottom up heapsort.
        Bottom-up heapify restores the heap order by first descending to a leaf using child comparisons, then inserting the displaced root element at the highest position that preserves the heap invariant.
        bottom-up heapify more efficient than classical sift-down, because value is moved only once, instead of swapping repeatedly at each level.
        """
        root = i    # i is a node that violates the heap property (not the correct order...)
        value = input_array[root]

        # * traverse the heap...
        while True:
            left = 2 * root + 1 # identify left child
            # * Exit Condition: - reached a leaf. no more nodes to search through.
            if left >= heap_size:
                break
            right = left + 1  # identify right child.
            # determine the larger child. this will be moved into the root (i) place
            if right < heap_size and input_array[right] > input_array[left]:
                child = right
            else:
                child = left
            # * copies the larger child of the root into the root node.
            input_array[root] = input_array[child]
            root = child    # copies the former root into the child position.

        # * traverse back up heap (from the former root position - to the root node.)
        while root > i:
            parent = (root - 1) // 2    # identifies the parent index
            # * exit condition -- parent is greater than or equal to the displaced value, placing value here preserves the max-heap property.
            if input_array[parent] >= value:
                break
            # * swaps the parent and the former root.
            input_array[root] = input_array[parent]
            root = parent
        # * places the displaced value into its final correct position (former root position)
        input_array[root] = value

    def _recursive_floyd_heapify(self):
        """
        uses recursion instead of iteration
        https://www.geeksforgeeks.org/dsa/heap-sort/
        """
        pass

    def _floyd_heapify(self, input_array, i, heap_size):
        """
        An operation that restores the heap property for a subtree whose root may violate the heap invariant, assuming its children are already valid heaps.
        Enforces parent ≥ children (max-heap) or parent ≤ children (min-heap).
        Precondition: Left and right subtrees of index i are already heaps. Only the node at i may be out of place. (the root)
        Postcondition: The subtree rooted at i becomes a valid heap.
        """
        while True:
            largest = i # current root - largest element in a max heap.
            # identify children (via heap index property)
            left = 2 * i + 1
            right = 2 * i + 2

            # * “Check the left and right children. If either child is bigger than the root, record its index as largest so we can swap it up and preserve the max-heap property.”
            if left < heap_size and input_array[left] > input_array[largest]:
                largest = left
            if right < heap_size and input_array[right] > input_array[largest]:
                largest = right
            # * Exit Condition: root is already the largest element in the current subheap.
            if largest == i:
                break
            # * swap current root with the largest child.
            input_array[i], input_array[largest] = input_array[largest], input_array[i]
            i = largest

    # --------------- Sort Operations ---------------
    def floyd_sort(self, input_array) -> Iterable:
        """user facing method - for heapsort implementations"""

        array_length = len(input_array)

        # * 1.) build max-heap
        # array_length // 2 - 1 is the last internal node in a 0-indexed heap.
        # loop goes backwards to the root.
        for i in range(array_length // 2 - 1, -1, -1):
            self._floyd_heapify(input_array, i, array_length)

        # * 2.) Each iteration removes the maximum element (root of the heap) and moves it to the sorted portion at the end.
        # end represents the current last index of the heap.
        for root in range(array_length -1, 0, -1):
            input_array[0], input_array[root] = input_array[root], input_array[0]
            self._floyd_heapify(input_array, 0, root)

        return input_array

    def bottom_up_sort(self, input_array) -> Iterable:
        """
        bottom up heapsort is a variant that reduces the number of comparisons required by a significant factor
        Bottom-up heapsort does not change asymptotic complexity; it reduces constant factors by minimizing swaps and comparisons.
        """
        array_length = len(input_array)

        # * 1.) build max-heap (Floyd, bottom-up)
        # array_length // 2 - 1 is the last internal node in a 0-indexed heap.
        # loop goes backwards to the root.
        for i in range(array_length // 2 - 1, -1, -1):
            self._bottom_up_heapify(input_array, i, array_length)

        # * 2.) Each iteration removes the maximum element (root of the heap) and moves it to the sorted portion at the end.
        # end represents the current last index of the heap.
        for end in range(array_length - 1, 0, -1):
            input_array[0], input_array[end] = input_array[end], input_array[0]
            self._bottom_up_heapify(input_array, 0, end)

        return input_array

    def out_of_place_sort(self, input_array) -> VectorArray:
        """
        Out-of-place heapsort using a -∞ sentinel:
        Standard bottom-up heapsort may still have some worst-case paths where a value repeatedly “bounces” up and down, especially with large keys near the leaves.
        Out-of-place heapsort avoids these repeated comparisons by using a separate output array. It guarantees n log₂ n + O(n) comparisons & Reduces unnecessary swaps and upward bounces
        Can be used as a building block for in-place QuickHeapsort
        """

        array_length = len(input_array)
        # new_heap: A copy of the input array. The heap operations are performed here so the input array remains unchanged.
        new_heap = input_array.copy()
        # Out-of-place means the sorted results are stored in a separate array (VectorArray) rather than rearranging the original array.
        sorted_results = VectorArray(array_length, object)
        reverse_results = VectorArray(array_length, object)

        # * 1.) Build Max Heap: (loops backwards)
        last_internal_node = array_length // 2 - 1
        for i in range(last_internal_node, -1, -1):
            # Uses bottom-up heapify, which is more efficient (O(N)) than classic sift-down because each displaced value is moved once instead of repeatedly swapped.
            self._bottom_up_heapify(new_heap, i, array_length)

        # * 2.) Extract max value repeatedly
        heap_size = array_length
        for _ in range(array_length):
            max_value = new_heap[0] # the root is the largest value
            sorted_results.append(max_value)  # append to sorted array.
            heap_size -= 1  # Reduce effective heap size (we have "logically" removed the root from the heap.)
            if heap_size > 0:
                # Pick the last element (displaced_value) to replace the root.
                displaced_value = new_heap[heap_size]
                # Optionally set new_heap[heap_size] = -inf as a sentinel (it’s outside the active heap, so it won’t affect extraction).
                new_heap[heap_size] = float("-inf")
                # Place the displaced value at the root (new_heap[0]).
                new_heap[0] = displaced_value
                # _bottom_up_heapify moves the displaced value down to the correct leaf and then back up to its proper position if needed. (notice how it operates on the heap size and not the full length of the array.)
                self._bottom_up_heapify(new_heap, 0, heap_size)

        # * reverse the results so they are in ascending order.
        for i in reversed(sorted_results):
            reverse_results.append(i)
        return reverse_results

    # --------------- Strategy Pattern ---------------
    def sort(self, input_array) -> Optional[Iterable]:
        """strategy pattern for heap sort types."""
        if self._technique == 'bottom_up':
            return self.bottom_up_sort(input_array)
        elif self._technique == 'floyd':
            return self.floyd_sort(input_array)
        elif self._technique == 'out_of_place':
            return self.out_of_place_sort(input_array)
        else:
            raise DsTypeError(f"Error: Invalid Heap Sort type chosen, ensure that technique is a valid heapsort technique.")


# ------------------------------- Main: Client Facing Code: -------------------------------

def main():

    print(f"\nTesting floyd heapsort")
    test_array = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(test_array)
    floyd = HeapSort().sort(test_array)
    print(floyd)

    print(f"\nTesting bottom up heapsort")
    test_array_2 = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(test_array_2)
    bottom_up = HeapSort(technique="bottom_up").sort(test_array_2)
    print(bottom_up)

    print(f"\nTesting out of place heapsort")
    test_array_3 = [3245,543,765,86797,89786,780,68467,57,6,345,345,32,534,25,24547,68,769,86,34235,3,1,1,5246,34,7546,8675,76,895,6643,15,34,654,867,987,97,8,32,362,4567,4568,9,67,976,743]
    print(test_array_3)
    out_of_place = HeapSort(technique="out_of_place").sort(test_array_3)
    print(out_of_place)

if __name__ == "__main__":
    main()
