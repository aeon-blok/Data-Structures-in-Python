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

# endregion

# region custom imports
from utils.helpers import RandomClass
from user_defined_types.generic_types import T
from utils.constants import DLL_SEPERATOR
from utils.validation_utils import DsValidation
from utils.exceptions import *
from utils.representations import PlistRepr


from adts.collection_adt import CollectionADT
from adts.positional_list_adt import PositionalListADT, iNode, iPosition

from ds.primitives.Positional_Lists.position import PNode, Position
from ds.primitives.Positional_Lists.positional_list_utils import PositionalListUtils

# endregion


class PositionalList(PositionalListADT[T], CollectionADT):
    """
    abstracted dll that uses position objects instead of nodes for references.
    Implemented with Sentinels - returns sentinel values for the underflow / overflow (None)
    """
    def __init__(self, datatype: type) -> None:
        self._header = PNode()
        self._trailer = PNode()
        self._header.next = self._trailer   # sentinel
        self._trailer.prev = self._header   # sentinel
        self._total_nodes = 0
        self._datatype = datatype
        # composed objects
        self._validators = DsValidation()
        self._utils = PositionalListUtils(self)
        self._desc = PlistRepr(self)

    @property
    def datatype(self):
        return self._datatype
    
    @property
    def total_nodes(self):
        return self._total_nodes
    
    @property
    def head(self):
        return self.first()
    
    @property
    def tail(self):
        return self.last()

    # ----- Utilities -----
    def __getitem__(self, key: Optional[iPosition[T]]) -> T:
        # if the key is a node - just return the element. (o(1))
        if isinstance(key, iPosition):
            target_node = key.node
            target_element = target_node.element
            return target_element
        else:
            raise KeyInvalidError("Error: Key needs to be a Node reference")

    def __setitem__(self, key: Optional[iPosition[T]], element: T) -> None:
        if isinstance(key, iPosition):
            replaced_value = self.replace(key, element) 
        else:
            raise KeyInvalidError("Error: Key needs to be a Node reference")

    def __str__(self) -> str:
        return self._desc.str_positional_list(DLL_SEPERATOR)

    def __repr__(self) -> str:
        return self._desc.repr_positional_list()

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        """check the length of the list"""
        return self._total_nodes

    def __contains__(self, element: Any) -> bool:
        """check if list contains an element value"""
        current_pos = self.first()
        while current_pos is not None:
            if current_pos.element == element:
                return True
            current_pos = self.after(current_pos)
        return False

    def is_empty(self) -> bool:
        """check if list is empty"""
        return self._total_nodes == 0

    def clear(self) -> None:
        """Clears and dereferences all Positions in the List."""
        current_pos = self.first()
        while current_pos is not None:
            # store info to move to next position
            next_pos = self.after(current_pos)
            # dereference and delete
            deleted_value = self.delete(current_pos)
            # traverse
            current_pos = next_pos

        self._total_nodes = 0

    def __iter__(self) -> Generator[T, None, None]:
        """Iterate through all the positions in the list and return the value"""
        # traverse
        return self._utils.positional_list_traversal()

    # ----- Accessor ADT Operations -----
    def first(self):
        """Access the Head Position"""
        node = self._utils.check_not_sentinel(self._header.next)
        return Position(node, container=self) if node else None

    def last(self):
        """Access the tail position"""
        node = self._utils.check_not_sentinel(self._trailer.prev)
        return Position(node, container=self) if node else None

    def before(self, position):
        """access the position before a specified position"""
        ref_node = self._utils.validate_position(position)
        previous_node = ref_node.prev
        # Start of List Case:
        if previous_node is self._header:
            return None
        return Position(previous_node, container=self)

    def after(self, position):
        """access the position after a specified position"""
        ref_node = self._utils.validate_position(position)
        next_node = ref_node.next
        # End of List Case:
        if next_node is self._trailer:
            return None
        return Position(next_node, container=self)

    def get(self, position):
        """retrieve the element value of a specified position"""
        ref_node = self._utils.validate_position(position)
        ref_value = ref_node.element
        return ref_value

    # ----- Mutator ADT Operations -----
    def add_first(self, element):
        """Add a position at the head"""
        self._validators.enforce_type(element, self.datatype)
        new_node = PNode(element, next=self._header.next, prev=self._header)
        relinked_node = self._utils.relink_nodes(new_node)
        self._total_nodes += 1  # update tracker
        return Position(relinked_node, self)

    def add_last(self, element):
        """Add a position at the tail."""
        self._validators.enforce_type(element, self.datatype)
        new_node = PNode(element, next=self._trailer, prev=self._trailer.prev)
        relinked_node = self._utils.relink_nodes(new_node)
        self._total_nodes += 1  # update tracker
        return Position(relinked_node, self)

    def add_before(self, position, element):  
        """ add a new position before a specified position reference"""      
        self._validators.enforce_type(element, self.datatype)
        ref_node = self._utils.validate_position(position)
        new_node = PNode(element, next=ref_node, prev=ref_node.prev)
        relinked_node = self._utils.relink_nodes(new_node)
        self._total_nodes += 1  # update tracker
        return Position(relinked_node, self)

    def add_after(self, position, element):
        """ add a new position after a specified position reference"""
        ref_node = self._utils.validate_position(position)
        self._validators.enforce_type(element, self.datatype)
        new_node = PNode(element, next=ref_node.next, prev=ref_node)
        relinked_node = self._utils.relink_nodes(new_node)
        self._total_nodes += 1  # update tracker
        return Position(relinked_node, self)

    def replace(self, position, element):
        """replace the element value of a specified position"""        
        old_node = self._utils.validate_position(position)
        self._validators.enforce_type(element, self.datatype)
        old_value = position.element
        old_node.element = element
        return old_value

    def delete(self, position): 
        """deletes a node at the specified position and returns the value"""
        # initialize nodes.
        old_node = self._utils.validate_position(position)
        old_position = position
        old_value = old_node.element
        previous_node = old_node.prev
        next_node = old_node.next

        # relink target node neighbours
        previous_node.next = next_node
        next_node.prev = previous_node

        # dereference old node
        old_position.container = None
        old_node.next = old_node.prev = None

        self._total_nodes -=1

        return old_value


# Main

def main():
    print("\n--- Initializing PositionalList ---")
    plist = PositionalList(str)
    print(f"Is empty? {plist.is_empty()}")

    # ---------- Normal Insertions ----------
    print("\n--- Insertions ---")
    pos0 = plist.add_last("0")
    print(plist)
    pos1 = plist.add_first("1")
    print(plist)
    pos2 = plist.add_last("2")
    print(plist)
    pos3 = plist.add_after(pos2, "3")
    print(plist)
    pos4 = plist.add_after(pos0, "4")
    print(plist)
    pos5 = plist.add_after(pos2, "5")
    print(plist)
    pos6 = plist.add_before(pos3, "6")
    print(plist)
    pos7 = plist.add_before(pos1, "7")
    print(plist)

    # ---------- __getitem__ Tests ----------
    print("\n--- Testing __getitem__ ---")
    print(f"Item at pos0: {plist[pos0]}")  # should print "0"
    print(f"Item at pos2: {plist[pos2]}")  # should print "2"
    print(f"Item at pos4: {plist[pos4]}")  # should print "4"

    # ---------- __setitem__ Tests ----------
    print("\n--- Testing __setitem__ ---")
    plist[pos1] = "100"
    print(f"After setting pos1 to '100': {plist[pos1]}")  # should print "100"

    plist[pos3] = "300"
    print(f"After setting pos3 to '300': {plist[pos3]}")  # should print "300"
    print(plist)

    # ---------- Deletions ----------
    print("\n--- Deletions ---")
    deleted_value = plist.delete(pos4)
    print(f"Deleted Node: {deleted_value}")
    print(plist)
    deleted_head = plist.delete(pos1)
    print(f"Deleted Head: {deleted_head}")
    print(plist)
    deleted_tail = plist.delete(pos3)
    print(f"Deleted Tail: {deleted_tail}")
    print(plist)

    # ---------- Iteration ----------
    print("\n--- Iteration ---")
    for i, item in enumerate(plist):
        print(f"Iterated over {i}: {item}")

    # ---------- Searches ----------
    print("\n--- Searches ---")
    print(plist.after(pos5))  # should return position after pos5
    print(plist.before(pos0))  # should return position before pos0
    print(plist)
    # for bidirectional search or by index, you would implement custom methods
    # here we just test positional navigation

    # ---------- Get/Set Item ----------
    print("\n--- Get/Set Item ---")
    print(f"Get item at pos0: {plist.get(pos0)}")
    plist.replace(pos0, "565")
    print(f"Set item at pos0: {plist.get(pos0)}")
    print(f"Get item at pos6: {plist.get(pos6)}")
    plist.replace(pos6, "6849")
    print(f"Set item at pos6: {plist.get(pos6)}")

    print(f"List length: {len(plist)}")
    print(f"Is '10' in PositionalList? {'10' in plist}")
    print(f"Is '200' in PositionalList? {'200' in plist}")
    print(repr(plist))

    print("\n--- Testing Error Cases ---")
    plist_b = PositionalList(str)
    another_list_pos = plist_b.add_first("A")

    # enforce_type errors
    try:
        plist.add_first(RandomClass("YOLO"))
    except Exception as e:
        print(f"Caught enforce_type error: {e}")

    try:
        plist.replace(pos6, RandomClass("YOLO"))
    except Exception as e:
        print(f"Caught enforce_type error: {e}")

    # validate_position errors
    fake_pos = "not_a_node"
    try:
        plist.add_after(fake_pos, "X")
    except Exception as e:
        print(f"Caught validate_position error: {e}")

    try:
        plist.delete(fake_pos)
    except Exception as e:
        print(f"Caught validate_position error: {e}")

    try:
        plist.delete(another_list_pos)
    except Exception as e:
        print(f"Caught Position belongs to another list error: {e}")

    # deleting from empty list
    # plist.clear()

    try:
        plist.delete(pos6)
    except Exception as e:
        print(f"Caught Deleted position Error: {e}")

    # ---------- Clear ----------
    print("\n--- Clearing List ---")
    plist.clear()
    print(plist)
    print(f"Is empty after clear? {plist.is_empty()}")
    print(repr(plist))

if __name__ == "__main__":
    main()
