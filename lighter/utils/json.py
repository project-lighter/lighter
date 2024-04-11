import json

from torch import Tensor


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            if obj.numel() > 100:
                return "Tensor of size {} too large to serialize".format(obj.size())
            return obj.cpu().tolist()
        # Let the base class default method raise the TypeError
        return super().default(obj)
