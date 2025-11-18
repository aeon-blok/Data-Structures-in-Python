import os


def count_loc_excluding_imports(root_dir: str):
    total_lines = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        # count only non-empty, non-comment, non-import lines
                        if (
                            stripped
                            and not stripped.startswith("#")
                            and not stripped.startswith("import")
                            and not stripped.startswith("from")
                        ):
                            total_lines += 1
    return total_lines


repo_path = r"J:\CODE\Python_Data_Structures_2025\src"
loc = count_loc_excluding_imports(repo_path)
print(f"LOC excluding imports: {loc}")
