from torch import nn


def replace_layer_with_identity(model, layer_name):
    """Can be used to replace any layer of the network with an Identity layer.
    Useful for removing the last layer of a network to be used as a backbone
    of an SSL model. 
    """
    setattr(model, layer_name, nn.Identity())
    return model


def remove_last_layer_sequentially(model):
    """Removes the last layer of a network and returns it as an nn.Sequential model.
    Useful when a network is to be used as a backbone of an SSL model.
    """
    return nn.Sequential(*list(model.children())[:-1])
