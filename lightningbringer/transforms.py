from copy import deepcopy


class Duplicate:

    def __init__(self, transforms1=None, transforms2=None):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, input):
        # Duplicate the input tensor
        out1, out2 = input, deepcopy(input)
        if self.transforms1 is not None:
            out1 = self.transforms1(out1)
        if self.transforms2 is not None:
            out2 = self.transforms2(out2)
        return (out1, out2)
