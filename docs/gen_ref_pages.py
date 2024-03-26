from pathlib import Path

import mkdocs_gen_files

PACKAGE = "lighter"
EXCLUDE = ["__pycache__"]

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / PACKAGE


def add_submodules_as_list(parent_folder: Path) -> str:
    """Add subfolders as a list to the index.md file."""
    python_files = []
    sub_folders = []
    for file in parent_folder.iterdir():
        if file.is_dir() and file.name not in EXCLUDE:
            sub_folders.append(file.name)
        elif file.suffix == ".py" and file.name != "__init__.py":
            python_files.append(file.name)

    output = []
    if len(sub_folders) > 0:
        output.append("\n## Submodules\n\n")
        for sub_folder_name in sub_folders:
            output.append(f"- [{sub_folder_name}]({sub_folder_name}/index.md)\n")

    if len(python_files) > 0:
        output.append("\n## Python Files\n\n")
        for python_file in python_files:
            python_link = python_file.replace(".py", ".md")
            output.append(f"- [{python_file}]({python_link})\n")

    return "".join(output)


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    module_py_notation = PACKAGE + "." + ".".join(module_path.parts)
    if module_py_notation in EXCLUDE:
        print(f"Excluding '{module_py_notation}' from the API reference.")
        continue
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = PACKAGE.split(".") + list(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        submodules_list = add_submodules_as_list(path.parent)
    elif parts[-1] == "__main__":
        continue
    else:
        submodules_list = ""
    nav[tuple(parts)] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")
        fd.write("\n\n" + submodules_list)
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
