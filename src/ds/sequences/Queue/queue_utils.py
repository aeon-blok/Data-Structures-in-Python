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
    Iterable,
    Type,
    TYPE_CHECKING,
)
from abc import ABC, ABCMeta, abstractmethod
from array import array
import numpy
import ctypes

# endregion


# region custom imports
from utils.exceptions import *


if TYPE_CHECKING:
    from utils.custom_types import T
    from adts.queue_adt import QueueADT
    from adts.collection_adt import CollectionADT


# endregion


class QueueUtils:
    def __init__(self, queue_obj) -> None:
        self.obj = queue_obj

    def check_empty_queue(self):
        """if the queue is empty raises an error."""
        if self.obj.is_empty():
            raise DsUnderflowError(f"Error: The Queue is empty.")


        
        
