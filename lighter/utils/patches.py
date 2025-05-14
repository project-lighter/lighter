"""
Contains code that patches certain issues from other libraries that we expect will be resolved in the future.
"""

from torch.nn import ModuleDict


class PatchedModuleDict(ModuleDict):
    """
    This class provides a workaround for key conflicts in PyTorch's ModuleDict by ensuring unique internal keys.
    """

    # https://github.com/pytorch/pytorch/issues/71203
    def __init__(self, modules=None):
        """
        Initializes the PatchedModuleDict with optional modules.

        Args:
            modules (dict, optional): A dictionary of modules to initialize the ModuleDict.
        """
        self._key_map = {}
        super().__init__(modules)

    def __setitem__(self, key, module):
        """
        Sets the module for the given key, ensuring a unique internal key.

        Args:
            key (str): The key to associate with the module.
            module (torch.nn.Module): The module to store.
        """
        internal_key = f"_{key}"
        while internal_key in self._modules:
            internal_key = f"_{internal_key}"
        self._key_map[key] = internal_key
        super().__setitem__(internal_key, module)

    def __getitem__(self, key):
        """
        Retrieves the module associated with the given key.

        Args:
            key (str): The key for which to retrieve the module.

        Returns:
            torch.nn.Module: The module associated with the key.
        """
        internal_key = self._key_map.get(key, key)
        return super().__getitem__(internal_key)

    def __delitem__(self, key):
        """
        Deletes the module associated with the given key.

        Args:
            key (str): The key for which to delete the module.
        """
        internal_key = self._key_map.pop(key, key)
        super().__delitem__(internal_key)

    def __contains__(self, key):
        """
        Checks if a module is associated with the given key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        internal_key = self._key_map.get(key, key)
        return super().__contains__(internal_key)

    def keys(self):
        """
        Returns the keys of the modules.

        Returns:
            KeysView: A view of the keys in the dictionary.
        """
        return self._key_map.keys()

    def items(self):
        """
        Returns the items (key, module) in the dictionary.

        Returns:
            Generator: A generator yielding key, module pairs.
        """
        return ((key, self._modules[internal_key]) for key, internal_key in self._key_map.items())

    def values(self):
        """
        Returns the modules in the dictionary.

        Returns:
            Generator: A generator yielding modules.
        """
        return (self._modules[internal_key] for internal_key in self._key_map.values())
