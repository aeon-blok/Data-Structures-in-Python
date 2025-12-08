"""





Import Hierarchy
user_defined_types → utils → adts → (primitives → sequences → maps → trees → graphs) → algorithms
make sure not to go backwards, will have to code around circular imports...



Composed Objects:
self._validators = Global validation methods wrapped in a helper class 
-- might have to scrap this now that most of the validation is being completed through UDT types.
self._utils = Utility Methods wrapped inside a composed helper class (per data structure)
self._desc = Used for Representations (how data structures look like in the console)


Custom Exceptions:
Have their own module -- they sit in the utils hierarchy

Key() objects:
right now hash tables return the key() objects - this is causing problems when used in graphs and so on
better for them to just return the value, maybe with a flag to return key() objects.
easier for the end user...

representations.py - __str__ for all data structures
validation_utils.py - validation logic for all data structures? -- replace this with validation in types.
User_Defined_Types/
  generic_types.py
  

exceptions.py
map_utils/
  hash_functions.py  (this includes compress func also)
  probe_functions.py
array_utils.py
linked_list_utils.py
tree_utils.py


All keys - should be the Key(input) class. This allows for validation of the keys.
base class is extendable.































"""
