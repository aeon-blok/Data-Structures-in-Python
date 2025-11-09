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

def str_node(self):
    pass

def repr_node(self):
    pass










# endregion


# stacks


# queues


# deques


# heaps


# Maps


# Trees


# Graphs
