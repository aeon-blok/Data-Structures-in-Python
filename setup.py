# setup.py - minimal working setup
from setuptools import setup, find_packages

setup(
    name="Python_Data_Structures_2025",  # Your project name
    version="0.1",  # Project version
    packages=find_packages(where="src"),  # Find all packages under src/
    package_dir={"": "src"},  # Tell Python packages are under src/
    install_requires=[],  # Optional: list dependencies here
)
