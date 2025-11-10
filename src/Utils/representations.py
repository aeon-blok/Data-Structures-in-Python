# where we add console visualizations for the different data structure types - usually use these in __str__ or __repr__ or a utility function.

# region arrays
def str_array(array_obj):
    """a list of strings representing all the elements in the array"""
    items = ", ".join(str(array_obj.array[i]) for i in range(array_obj.size))
    return f"[{array_obj.__class__.__qualname__}][{array_obj.datatype.__name__}][{array_obj.size}/{array_obj.capacity}][{items}]"

def repr_array(array_obj):
    """array __repr__ - for devs"""
    class_address = f"<{array_obj.__class__.__qualname__} object at {hex(id(array_obj))}>"
    data_type = f"Type: {array_obj.datatype.__name__}"
    storage = f"Capacity: {array_obj.size}/{array_obj.capacity}"
    array_type = f"Array Type: {'Static' if array_obj.is_static == True else 'Dynamic'}"
    return f"{class_address}, {data_type}, {storage}, {array_type}"

def str_view(view_obj):
    """ __str__ for array views (similar to slices in python without the copying)"""
    items = ", ".join(str(view_obj[i]) for i in range(view_obj._length))
    return f"[{view_obj.__class__.__qualname__}][{view_obj.datatype.__name__}][{view_obj._length}][{items}]"

def repr_view(view_obj):
    """ __repr__ for array views (like slices)"""
    class_address = f"<{view_obj.__class__.__qualname__} object at {hex(id(view_obj))}>"
    items = ", ".join(str(view_obj[i]) for i in range(view_obj._length))
    return f"{class_address}, Type: {view_obj.datatype.__name__}, Total Elements: {view_obj._length}"
# endregion

# region linked lists

def str_sll_node(sll_node_obj):
    node_element = f"{sll_node_obj.element}"
    return f"{node_element}"

def repr_sll_node(sll_node_obj):
    class_address = f"<{sll_node_obj.__class__.__qualname__} object at {hex(id(sll_node_obj))}>"
    next_pointer = f"Next Pointer: {str(sll_node_obj.next)}"
    node_element = f"Node: {sll_node_obj.element}"
    linked = f"Node in a list?: {sll_node_obj.is_linked}"
    owner = f"List Owner: {repr(sll_node_obj.list_owner)}"
    return f"{class_address}, {node_element}, {next_pointer}, {linked}, {owner}"


def str_sll(sll_obj, sep: str = " ->> "):
    """Displays all the content of the linked list as a string."""
    seperator = sep
    datatype = sll_obj.datatype.__name__
    total_nodes = sll_obj.total_nodes
    class_name = sll_obj.__class__.__qualname__

    if sll_obj._head is None:
        return f"[{class_name}][{datatype}][{total_nodes}]"

    def _simple_traversal():
        """traverses the nodes and returns a string via generator"""
        current_node = sll_obj._head
        while current_node:
            yield str(current_node._element)
            current_node = current_node.next

    infostring = f"[{class_name}][{datatype}][{total_nodes}]: (H) {seperator.join(_simple_traversal())} (T)"
    return infostring


def repr_sll(sll_obj):
    class_address = f"<{sll_obj.__class__.__qualname__} object at {hex(id(sll_obj))}>"
    datatype = sll_obj.datatype.__name__
    total_nodes = sll_obj.total_nodes
    return f"{class_address}, Type: {datatype}, Total Nodes: {total_nodes}"


# endregion


# stacks


# queues


# deques


# heaps


# Maps


# Trees


# Graphs
