import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift

def calculate_cross_correlation(original_line, shifted_line):
    """Calculate cross-correlation between two 1D arrays."""
    cross_corr = np.correlate(shifted_line, original_line, mode='full')
    return cross_corr

def detect_shifts(image, shifted_image):
    """Detect horizontal and vertical shifts using cross-correlation."""
    # Choose representative row and column
    row_idx = image.shape[0] // 2  # Middle row
    col_idx = image.shape[1] // 2  # Middle column

    # Horizontal shift (row-based)
    horizontal_corr = calculate_cross_correlation(image[row_idx, :], shifted_image[row_idx, :])
    horizontal_shift = np.argmax(horizontal_corr) - (len(horizontal_corr) // 2)

    # Vertical shift (column-based)
    vertical_corr = calculate_cross_correlation(image[:, col_idx], shifted_image[:, col_idx])
    vertical_shift = np.argmax(vertical_corr) - (len(vertical_corr) // 2)

    return vertical_shift, horizontal_shift

def realign_image(shifted_image, vertical_shift, horizontal_shift):
    """Realign the image by reversing the detected shifts."""
    return shift(shifted_image, shift=(-vertical_shift, -horizontal_shift))

# Load images
image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")

# Detect shifts
vertical_shift, horizontal_shift = detect_shifts(image, shifted_image)
print(f"Detected Shifts: Vertical = {vertical_shift}, Horizontal = {horizontal_shift}")

# Realign the image
reversed_shifted_image = realign_image(shifted_image, vertical_shift, horizontal_shift)

# Plot the results
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Shifted Image
plt.subplot(2, 3, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title("Shifted Image")
plt.axis('off')

# Reversed Shifted Image
plt.subplot(2, 3, 3)
plt.imshow(reversed_shifted_image, cmap='gray')
plt.title("Reversed Shifted Image")
plt.axis('off')

plt.tight_layout()
plt.show()
