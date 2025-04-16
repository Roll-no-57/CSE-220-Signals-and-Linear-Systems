import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load and preprocess the image
image = plt.imread('noisy_image.png')

# Convert to grayscale if needed
if image.ndim == 3:
    image = np.mean(image, axis=2)

# Normalize image
image = image / 255.0
print("Image shape:", image.shape)

rows, cols = image.shape
denoised_image = np.zeros_like(image)

# Sample rate (arbitrary for image rows)
sample_rate = 1000
sampled_times = np.linspace(0, 1, cols)
max_freq = sample_rate / 2
frequencies = np.linspace(0, max_freq, cols)


# Custom FT using trapezoidal integration
def custom_fourier_transform(signal, frequencies, times):
    num_freqs = len(frequencies)
    ft_real = np.zeros(num_freqs)
    ft_imag = np.zeros(num_freqs)

    for i, f in enumerate(frequencies):
        cos_term = np.cos(2 * np.pi * f * times)
        sin_term = np.sin(2 * np.pi * f * times)
        ft_real[i] = np.trapezoid(signal * cos_term, times)
        ft_imag[i] = -1 * np.trapezoid(signal * sin_term, times)

    return ft_real, ft_imag


# Custom IFT using trapezoidal integration
def custom_inverse_fourier_transform(ft_signal, frequencies, times):
    reconstructed = np.zeros(len(times))

    for i in range(len(times)):
        cos_term = np.cos(2 * np.pi * frequencies * times[i])
        sin_term = np.sin(2 * np.pi * frequencies * times[i])
        reconstructed[i] = np.trapezoid(ft_signal[0] * cos_term - ft_signal[1] * sin_term, frequencies)

    return reconstructed


# Step 2–4: Process each row
for i in range(rows):
    row = image[i, :]

    # FT
    ft_real, ft_imag = custom_fourier_transform(row, frequencies, sampled_times)

    # Step 3: Filtering — keep only low frequencies
    cutoff_freq = 200  # Hz, adjust this experimentally
    filtered_real = ft_real.copy()
    filtered_imag = ft_imag.copy()

    filtered_real[frequencies > cutoff_freq] = 0
    filtered_imag[frequencies > cutoff_freq] = 0

    # IFT
    reconstructed_row = custom_inverse_fourier_transform((filtered_real, filtered_imag), frequencies, sampled_times)

    # Store denoised row
    denoised_image[i, :] = reconstructed_row

# Clip any overshoot (can happen due to reconstruction)
denoised_image = np.clip(denoised_image, 0, 1)

# Step 5: Save and show
plt.imsave('denoised_image.png', denoised_image, cmap='gray')

plt.figure()
plt.title('Denoised Image (Custom FT)')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')
plt.show()
