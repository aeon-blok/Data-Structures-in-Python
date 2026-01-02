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
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes
import random
from collections.abc import Sequence

# endregion


# region custom imports
from user_defined_types.generic_types import T, K, Index
from user_defined_types.key_types import iKey, Key


# endregion


"""
Matrix ADT:
a fixed sized grid of elements arranged in rows & columns used to represent and manipulate linear relationships
While you can store anything in a matrix, most of the mathematical methods in the ADT (like get_determinant, add_Matrices, or invert_matrix) only make sense if T is a number or a symbolic variable.
"""

class MatrixADT(ABC, Generic[T]):
    """Matrix ADT Interface"""

    # ----- Canonical ADT Operations -----
    # ----- Accessor ADT Operations -----
    @abstractmethod
    def num_rows(self) -> int:
        """returns the total number of rows in the matrix"""
        ...

    @abstractmethod
    def num_columns(self) -> int:
        """returns the total number of columns in the matrix"""
        ...

    @abstractmethod
    def get_element(self, row: Index, column: Index) -> Optional[T]:
        """returns the element stored at the cell (row, col)"""
        ...

    @abstractmethod
    def is_equal(self, matrix: 'MatrixADT[T]') -> bool:
        """Compares both matrices. If they have the same shape and values returns True"""
        ...

    @abstractmethod
    def is_zero_matrix(self) -> bool:
        """returns true if matrix only has 0 for every element."""
        ...

    @abstractmethod
    def is_identity_matrix(self) -> bool:
        """
        returns true if and only if the matrix is square AND has 1's element values aloing its central diagonal (top left -> bottom right)
        every other element value must be 0
        """
        ...

    @abstractmethod
    def get_trace(self) -> T:
        """returns the accumulated sum of all the central diagonal elements (top left -> bottom right)"""
        ...

    @abstractmethod
    def get_determinant(self) -> T:
        """
        returns the scaling factor & orientation. 
        This tells you how much the matrix transformation scales the matrix space
        It also explains the orientation - if the orientation is preserved, or reversed.
        a matrix cannot be inverted if the result is 0.
        """
        ...

    # ----- Mutator ADT Operations -----

    # Factory Generators
    @abstractmethod
    def create_zero_matrix(self, rows: int, columns: int) -> 'MatrixADT[T]':
        """creates a matrix with the given dimensions, and fills every cell with 0"""
        ...

    @abstractmethod
    def create_identity_matrix(self, size: int) -> "Optional[MatrixADT[T]]":
        """creates a Square matrix. the central diagonal is filled with 1's, every other element is 0 """
        ...

    @abstractmethod
    def create_matrix_from_rule(self, rows: int, columns: int, formula: Callable) -> 'Optional[MatrixADT[T]]':
        """Creates a Matrix where each cell element value is computed from a specified formula."""
        ...

    @abstractmethod
    def set_element(self, row: Index, column: Index, element: T) -> None:
        """replaces or sets the element value for a cell in the matrix."""
        ...

    @abstractmethod
    def transpose_matrix(self) -> 'Optional[MatrixADT[T]]':
        """Flips the Matrix so that the rows become columns, and the columns become rows"""
        ...

    @abstractmethod
    def add_matrices(self, matrix: 'MatrixADT[T]') -> 'Optional[MatrixADT[T]]':
        """ Creates a New Matrix by adding the corresponding elements of two matrices that share the same shape (same number of rows and columns)"""
        ...

    @abstractmethod
    def subtract_matrices(self, matrix: 'MatrixADT[T]') -> 'Optional[MatrixADT[T]]':
        """same as add, but uses subtraction for each element. the original matrix is not modified"""

    @abstractmethod
    def negate_matrix(self) -> "Optional[MatrixADT[T]]":
        """multiplies every element in the matrix by -1"""
        ...

    @abstractmethod
    def scale_matrix(self, scalar: T) -> "Optional[MatrixADT[T]]":
        """Multiplies every element by the given scalar input."""
        ...

    @abstractmethod
    def multiply_matrices(self, matrix: 'MatrixADT[T]') -> "Optional[MatrixADT[T]]":
        """merges 2 matrices together using matrix multiplication"""
        ...

    @abstractmethod
    def invert_matrix(self) -> "Optional[MatrixADT[T]]":
        """If Possible: reverses the effect of a previous transformation to this matrix."""
        ...

    @abstractmethod
    def extract_sub_matrix(self, row_start: Index, row_end: Index, column_start: Index, column_end: Index) -> "Optional[MatrixADT[T]]":
        """creates a new sub matrix from a specified range of the original matrix."""
        ...
