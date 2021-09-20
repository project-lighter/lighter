import inspect
import pkgutil
from pathlib import Path


def import_class_from_dirs_and_modules(class_name, dirs_modules):
    """Looks for a particular class of `class_name` recursively inside of the
    given modules and directories. The directories that are given to it need
    to have __init__.py in order for this function to recognize its modules."""

    # Goes through the given dirs and modules and gets their absolute paths
    paths = []
    for path in dirs_modules:
        if inspect.ismodule(path):
            dir_path = Path(path.__file__).parent
        else:
            dir_path = Path(path).resolve()
        paths.append(str(dir_path))

    # Goes through the parsed module and dir paths recursevily
    for loader, module_name, _ in pkgutil.walk_packages(paths):
        # Loads a particular module
        module = loader.find_module(module_name).load_module(module_name)
        # Goes through the attributes of the module and checks if it
        # is the one being sought, and if it, returns it.
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if inspect.isclass(attribute) and class_name == attribute_name:
                return attribute

    raise ValueError(
        f"Class with name `{class_name}`` not found in any of the given directories or modules. "
        "If it is located in a project folder, set `project_dir` in config as the project's path.")
