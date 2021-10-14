import torch


def min_max_normalize(image, min_value, max_value):
    image = image.float()
    image = (image - min_value) / (max_value - min_value)
    return 2 * image - 1


def min_max_denormalize(image, min_value, max_value):
    image += 1
    image /= 2
    image *= (max_value - min_value)
    image += min_value
    return image
