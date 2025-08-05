import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def spectral_bias(f, beta, sigma=5):
    """
    Compute spectral bias using Method 2 (2D convolution-based) with detailed visualization and explanation.

    Parameters:
    f (numpy array): Target function (image).
    beta (numpy array): Neural network approximation of f.
    sigma (float): Standard deviation for Gaussian filter to approximate the sinc function.

    Returns:
    float: Spectral bias.
    """
    # Step 1: Compute Residual Image
    r = f - beta  # The difference between the ground truth and the model's prediction

    # Step 2: Apply 2D Gaussian filter as a low-pass filter to extract low-frequency components
    r_low = gaussian_filter(r, sigma=sigma)  # Smooths the residual to retain only low-frequency components

    # Step 3: Compute the high-frequency component by subtracting low-frequency from original residual
    r_high = r - r_low  # This isolates the high-frequency details (edges, fine textures)

    # Step 4: Compute energy (variance) in each frequency range
    total_variance = np.sum(f ** 2)  # Total signal energy in the target image
    Elow = np.sum(r_low ** 2) / total_variance  # Energy in low-frequency residual
    Ehigh = np.sum(r_high ** 2) / total_variance  # Energy in high-frequency residual

    SB = (Ehigh - Elow) / (Ehigh + Elow)
    return np.abs(SB)