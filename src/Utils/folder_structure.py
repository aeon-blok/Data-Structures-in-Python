from pathlib import Path


IGNORE = {"__pycache__", "Python_Data_Structures_2025.egg-info"}

def tree(root: Path, prefix=""):
    entries = [
        p for p in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
        if p.name not in IGNORE
    ]
    for i, p in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + p.name)
        if p.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            tree(p, prefix + extension)


tree(Path(r"J:\CODE\Python_Data_Structures_2025\src"))
