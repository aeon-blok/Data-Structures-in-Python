import _collections_abc
from collections import Counter

# just random testing..... delete later

# Collections Module:

# Counter:
# stores elements as dict keys, and their count as the value.

# iterated over a string.
string = "BoisterousBoisterous"
new_counter = Counter(string)
print(new_counter)
print(new_counter.keys())
print(new_counter.values())

new_counter.clear()

# this will map the elements with default counter values.
element_map = {"apple": 5, "banana": 3, "cherry": 6}
new_counter = Counter(element_map)
print(new_counter)


# prints the most common element and is acessed in descending order from most common. (1 - most common)
print(new_counter.most_common(1))

# elements()
print(list(new_counter.elements()))
