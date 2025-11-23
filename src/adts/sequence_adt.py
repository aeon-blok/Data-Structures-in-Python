# region standard imports
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
    Iterable
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
# endregion

# region custom imports
from user_defined_types.generic_types import T, Index

# endregion


"""
Sequence ADT: 
models an ordered, finite collection of elements, each accessible by an integer position. 

Properties:
Sequential ordering: Iteration returns items in insertion/logical order.
Deterministic behavior: Given same state + same operation ⇒ same state result.
Referential transparency at value level: Reading doesn't change the sequence.
Clone semantics: Copy produces a new sequence with identical elements and order.
Isolation of mutation: Only the targeted index region is affected by set, insert, delete.

Constraints:
Homogeneous or generic: All elements conform to the type parameter T (generic type).
Atomicity (abstract): Each operation is indivisible at conceptual level.
Mutable: Update and structural mutation supported (set, insert, delete).

Invariants:
Order Preservation: Sequence preserves element order: a before b  ⇒  remains before b -- unless modified by defined ops
Stable positions (until mutation): Index meaning persists between operations unless an edit shifts structure.
Finite cardinality: Sequence length n is finite and well-defined.
Positional identity: An element’s identity is tied to its index, not its value.
Index Domain: Valid indices always form a contiguous range
Shift semantics: Insert shifts right; delete shifts left, preserving order.
No implicit reordering: No automatic sorting or rearrangement.
Stable traversal: Traversal always yields exactly size(S) elements.
Totality (No “holes”): Never exists an undefined slot between elements.
Structural Consistency: After any operation, invariants must still hold.
Size Correspondence: The length of the sequence is tied to the domain of the defined indices
"""


# interface
class SequenceADT(ABC, Generic[T]):
    """Sequence ADT: models an ordered, finite collection of elements, each accessible by an integer position (index)."""

    # ----- Utility ADT Operations -----
    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        """Iterates over all the elements in the sequence - used in loops and ranges etc"""
        pass

    # ----- Canonical ADT Operations -----
    @abstractmethod
    def get(self, index: Index) -> T:
        """Return element at index i"""
        pass

    @abstractmethod
    def set(self, index: Index, value: T) -> None:
        """Replace element at index i with value"""
        pass

    @abstractmethod
    def insert(self, index: Index, value: T) -> None:
        """Insert value at index i, shift elements right"""
        pass

    @abstractmethod
    def delete(self, index: Index) -> T:
        """Remove element at index i, shift elements left"""
        pass

    @abstractmethod
    def append(self, value: T) -> None:
        """Add value at end N-1"""
        pass

    @abstractmethod
    def prepend(self, value: T) -> None:
        """Insert value at index 0"""
        pass

    @abstractmethod
    def index_of(self, value: T) -> Optional[int]:
        """Return index number of first value (if exists)"""
        pass
