"""





Import Hierarchy
utils → adts → (primitives → sequences → maps → trees → graphs) → algorithms
make sure not to go backwards, will have to code around circular imports...



Composed Objects:
self._validators = Global validation methods wrapped in a helper class
self._utils = Utility Methods wrapped inside a composed helper class (per data structure)
self._desc = Used for Representations (how data structures look like in the console)


Custom Exceptions:
Have their own module -- they sit in the utils hierarchy



representations.py - __str__ for all data structures
validation_utils.py - validation logic for all data structures?
exceptions.py
custom_types.py
hash_utils/
  hash_functions.py
  compression.py
  probe_functions.py
array_utils.py
linked_list_utils.py
tree_utils.py

































"""
