from copy import deepcopy
from typing import Any, Callable, Tuple, Union

import torch


class Duplicate:

    def __init__(self,
                 transforms1: Optional[Callable] = None,
                 transforms2: Optional[Callable] = None):
        """Duplicates an input and applies the given transformations to each copy separately.

        Args:
            transforms1 (Callable, optional): transforms to apply over the first copy of the
                input. Defaults to None.
            transforms2 (Callable, optional): transforms to apply over the second copy of the
                input. Defaults to None.
        """
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of two tensors.
        """


        """
        Returns:
            : Color jittered image.
        """
        out1, out2 = input, deepcopy(input)
        if self.transforms1 is not None:
            out1 = self.transforms1(out1)
        if self.transforms2 is not None:
            out2 = self.transforms2(out2)
        return (out1, out2)
