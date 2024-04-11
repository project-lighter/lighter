from functools import lru_cache

import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import color


@lru_cache(maxsize=128)
def distinguishable_colors(n_colors, bg: np.ndarray | None = None, return_as_uint8: bool = False) -> np.ndarray:
    """
    Generates a set of n_colors that are maximally perceptually distinct.

    Args:
        n_colors (int): The number of desired colors.
        bg (list, optional): The background color(s) in RGB format. Values should be in the range [0, 1].
            If None, white is used as the background color. Defaults to None.
        return_as_uint8 (bool, optional): If True, the output colors are in the range [0, 255]. Defaults to False.
    Returns:
        ndarray: An n_colors x 3 array containing the RGB values of the distinguishable colors. Values are in the range [0, 1].

    Raises:
        ValueError: If the requested number of colors cannot be distinguished effectively.
    """

    # Handle background color(s)
    if bg is None:
        bg_rgb = np.array([[1, 1, 1]])
    elif not isinstance(bg, np.ndarray):
        bg_rgb = np.array(bg)
    else:
        bg_rgb = bg
    if bg_rgb.ndim == 1:
        bg_rgb = np.expand_dims(bg_rgb, axis=0)
    if np.max(bg_rgb) > 1 or np.min(bg_rgb) < 0:
        raise ValueError("Color values should be in the range [0, 1].")

    # Generate a large number of candidate RGB colors
    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.vstack((R.ravel(), G.ravel(), B.ravel())).T

    # Check if enough colors can be distinguished
    if n_colors > rgb.shape[0]:
        raise ValueError("Cannot distinguish this many colors effectively.")

    # Convert colors to Lab space
    lab = color.rgb2lab(rgb)
    bg_lab = color.rgb2lab(bg_rgb)

    # Calculate distances to background colors
    min_dist2 = np.inf * np.ones([rgb.shape[0]])
    for bg_color in bg_lab:
        dist2 = np.sum((lab - bg_color) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    # Iteratively pick the most distant color
    colors = np.zeros((n_colors, 3))
    last_lab = bg_lab[-1, :]  # Initialize with last background color
    for i in range(n_colors):
        dist2 = np.sum((lab - last_lab) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        farthest_idx = np.argmax(min_dist2)
        colors[i] = rgb[farthest_idx]
        last_lab = lab[farthest_idx]

    if return_as_uint8:
        colors = (colors * 255).round().astype(np.uint8)
    return colors
