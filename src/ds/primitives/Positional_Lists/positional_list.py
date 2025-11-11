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
from utils.custom_types import T
from utils.constants import DLL_SEPERATOR
from utils.validation_utils import enforce_type, index_boundary_check
from utils.representations import repr_positional_list, str_positional_list
from utils.positional_list_utils import validate_position, make_position, insert_between, positional_list_traversal
from adts.collection_adt import CollectionADT
from adts.positional_list_adt import PositionalListADT, iNode, iPosition
from ds.primitives.Positional_Lists.position import PNode, Position

# endregion


class PositionalList(PositionalListADT[T], CollectionADT):
    """
    abstracted dll that uses position objects instead of nodes for references.
    """
    def __init__(self, datatype: type) -> None:
        self._header = PNode()
        self._trailer = PNode()
        self._header.next = self._trailer   # sentinel
        self._trailer.prev = self._header   # sentinel
        self._total_nodes = 0
        self._datatype = datatype

    @property
    def datatype(self):
        return self._datatype
    @property
    def total_nodes(self):
        return self._total_nodes

    # ----- Utilities -----

    def __getitem__(self, key: Optional[iPosition[T]]) -> T:
        # if the key is a node - just return the element. (o(1))
        if isinstance(key, iPosition):
            target_node = key.node
            target_element = target_node.element
            return target_element
        else:
            raise TypeError("Error: Key needs to be a Node reference")

    def __setitem__(self, key: Optional[iPosition[T]], element: T) -> None:
        if isinstance(key, iPosition):
            replaced_value = self.replace(key, element) 
        else:
            raise TypeError("Error: Key needs to be a Node reference")

    def __str__(self) -> str:
        return str_positional_list(self, DLL_SEPERATOR)

    def __repr__(self) -> str:
        return repr_positional_list(self)

    # ----- Meta Collection ADT Operations -----
    def __len__(self) -> int:
        """check the length of the list"""
        return self._total_nodes

    def __contains__(self, element: Any) -> bool:
        """ check if list contains an element value"""
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
        yield from positional_list_traversal(self)

    # ----- Accessor ADT Operations -----
    def first(self):
        """Access the Head Position"""
        return make_position(self, self._header.next)

    def last(self):
        """Access the tail position"""
        return make_position(self, self._trailer.prev)

    def before(self, position):
        """access the position before a specified position"""
        ref_node = validate_position(self, position)
        previous_node = ref_node.prev
        # Start of List Case:
        if previous_node is self._header:
            return None
        return make_position(self, previous_node)

    def after(self, position):
        """access the position after a specified position"""
        ref_node = validate_position(self, position)
        next_node = ref_node.next
        # End of List Case:
        if next_node is self._trailer:
            return None
        return make_position(self, next_node)

    def get(self, position):
        """retrieve the element value of a specified position"""
        ref_node = validate_position(self, position)
        ref_value = ref_node.element
        return ref_value

    # ----- Mutator ADT Operations -----
    def add_first(self, element):
        """Add a position at the head"""
        enforce_type(element, self.datatype)
        return insert_between(self, element, previous_node=self._header, next_node=self._header.next)

    def add_last(self, element):
        """Add a position at the tail."""
        enforce_type(element, self.datatype)
        return insert_between(self, element, previous_node=self._trailer.prev, next_node=self._trailer)

    def add_before(self, position, element):  
        """ add a new position before a specified position reference"""      
        enforce_type(element, self.datatype)
        ref_node = validate_position(self, position)
        return insert_between(self, element, previous_node=ref_node.prev, next_node=ref_node)

    def add_after(self, position, element):
        """ add a new position after a specified position reference"""
        ref_node = validate_position(self, position)
        enforce_type(element, self.datatype)
        return insert_between(self, element, previous_node=ref_node, next_node=ref_node.next)

    def replace(self, position, element):
        """replace the element value of a specified position"""        
        old_node = validate_position(self, position)
        enforce_type(element, self.datatype)
        old_value = position.element
        old_node.element = element
        return old_value

    def delete(self, position): 
        """deletes a node at the specified position and returns the value"""
        # initialize nodes.
        old_node = validate_position(self, position)
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

    # ---------- Clear ----------
    print("\n--- Clearing List ---")
    plist.clear()
    print(plist)
    print(f"Is empty after clear? {plist.is_empty()}")

    print("\n\n--- Testing Error Cases ---")
    plist = PositionalList(str)
    node = plist.add_first("A")

    # enforce_type errors
    try:
        plist.add_first(RandomClass("YOLO"))
    except Exception as e:
        print(f"Caught enforce_type error: {e}")

    try:
        plist.replace(node, RandomClass("YOLO"))
    except TypeError as e:
        print(f"Caught enforce_type error: {e}")

    # validate_position errors
    fake_node = "not_a_node"
    try:
        plist.add_after(fake_node, "X")
    except TypeError as e:
        print(f"Caught validate_position error: {e}")

    try:
        plist.delete(fake_node)
    except TypeError as e:
        print(f"Caught validate_position error: {e}")

    # deleting from empty list
    plist.clear()
    try:
        plist.delete(node)
    except ValueError as e:
        print(f"Caught Error: {e}")

if __name__ == "__main__":
    main()
