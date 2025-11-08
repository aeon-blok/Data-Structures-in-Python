import _collections_abc
from collections import Counter

# just random testing..... delete later


# todo seperate into its own module. can then import for all hash tables going forward instead of duplicating code....
"""
  hash_utils/
            hash_functions.py
            compression.py
            probing.py
            randomness.py
"""

# todo 


# imports hierarchy
# core → primitives → containers(or linear ds, or sequences) → maps → trees → graphs → algorithms

# setup.py - minimal working setup
from setuptools import setup, find_packages

setup(
    name="ds_refactor",  # Your project name
    version="0.1",  # Project version
    packages=find_packages(where="src"),  # Find all packages under src/
    package_dir={"": "src"},  # Tell Python packages are under src/
    install_requires=[],  # Optional: list dependencies here
)
