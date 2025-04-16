import numpy as np
import matplotlib.pyplot as plt


class Fourier:
    def __init__(self, func, func_name, x_values, frequencies):
        self.func = func
        self.func_name = func_name
        self.x_values = x_values
        self.y_values = func(x_values)
        self.sampled_times = x_values
        self.frequencies = frequencies

    # Fourier Transform
    def fourier_transform(self, signal, frequencies, sampled_times):
        num_freqs = len(frequencies)
        ft_result_real = np.zeros(num_freqs)
        ft_result_imag = np.zeros(num_freqs)

        # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
        # use trapezoidal integration to calculate the real and imaginary parts of the FT

        for i, f in enumerate(frequencies):
            cos_term = np.cos(2 * np.pi * f * sampled_times)
            sin_term = np.sin(2 * np.pi * f * sampled_times)
            ft_result_real[i] = np.trapezoid(signal * cos_term , sampled_times)
            ft_result_imag[i] = -1 * np.trapezoid(signal * sin_term, sampled_times)

        return ft_result_real, ft_result_imag

    # Inverse Fourier Transform
    def inverse_fourier_transform(self,ft_signal, frequencies, sampled_times):
        n = len(sampled_times)
        reconstructed_signal = np.zeros(n)
        # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
        # use trapezoidal integration to calculate the real part
        # You have to return only the real part of the reconstructed signal
        for i in range(n):
            cos_term = np.cos(2 * np.pi * frequencies * sampled_times[i])
            sin_term = np.sin(2 * np.pi * frequencies * sampled_times[i])
            reconstructed_signal[i] = np.trapezoid(ft_signal[0] * cos_term - ft_signal[1] * sin_term, frequencies)

        return reconstructed_signal


    def plot_original_function(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.x_values, self.y_values, label="Original y = " + self.func_name)
        plt.title("Original Function (y = " + self.func_name + ")")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def plot_frequency_spectrum(self, title=""):
        ft_data = self.fourier_transform(self.y_values, self.frequencies, self.sampled_times)
        plt.figure(figsize=(12, 6))
        plt.plot(self.frequencies, np.sqrt(ft_data[0] ** 2 + ft_data[1] ** 2))
        plt.title("Frequency Spectrum of y = " + self.func_name)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(title)
        plt.show()

    def frequesncy_spectrum(self):
        ft_data = self.fourier_transform(self.y_values, self.frequencies, self.sampled_times)
        return self.frequencies, np.sqrt(ft_data[0] ** 2 + ft_data[1] ** 2)

    def plot_reconstructed_function(self, title=""):
        ft_data = self.fourier_transform(self.y_values, self.frequencies, self.sampled_times)
        reconstructed_y_values = self.inverse_fourier_transform(ft_data, self.frequencies, self.sampled_times)
        plt.figure(figsize=(12, 4))
        plt.plot(self.x_values, self.y_values, label="Original y = " + self.func_name, color="blue")
        plt.plot(self.sampled_times, reconstructed_y_values, label="Reconstructed y = " + self.func_name , color="red", linestyle="--")
        plt.title("Original vs Reconstructed Function (y = " + self.func_name + ")")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(title)
        plt.show()


############################## MAIN FUNCTION ################################


# Defining functions
def parabolic_function(x):
    return np.where((x >= -2) & (x <= 2), x**2, 0)

def triangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1 - np.abs(x / 2), 0)

def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), (x + 2)/4, 0)

def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)
def given_function(t):
    return 2 * np.sin(14 * np.pi * t) - np.sin(2 * np.pi * t) * (4 * np.sin(2 * np.pi * t) * np.sin(14 * np.pi * t) - 1)


def main():

    # functions = [(parabolic_function, "x^2"), (triangular_function, "Triangular"), (sawtooth_function, "Sawtooth"),
    #              (rectangular_function, "Rectangular")]
    functions = [(given_function, "2sin(14πt) - sin(2πt)(4sin(2πt)sin(14πt) - 1)")]
    freqs = [(-1, 1), (-2, 2), (-5, 5)]

    for func, func_name in functions:
        x_values = np.linspace(-10, 10, 1000)

        frequencies = np.linspace(freqs[0][0], freqs[0][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_original_function()
        ft.plot_frequency_spectrum("Figure 1:  Frequency Spectrum of y = " + func_name + " (frequencies from " + str(
            freqs[0][0]) + " to " + str(freqs[0][1]) + ")")

        frequencies = np.linspace(freqs[1][0], freqs[1][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_frequency_spectrum("Figure 2: Frequency Spectrum of y = " + func_name + " (frequencies from " + str(
            freqs[1][0]) + " to " + str(freqs[1][1]) + ")")

        frequencies = np.linspace(freqs[2][0], freqs[2][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_frequency_spectrum("Figure 3: Frequency Spectrum of y = " + func_name + " (frequencies from " + str(
            freqs[2][0]) + " to " + str(freqs[2][1]) + ")")

        frequencies = np.linspace(freqs[0][0], freqs[0][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_reconstructed_function(
            "Figure 4: Reconstructed signal for frequency range " + str(freqs[0][0]) + " to " + str(freqs[0][1]))

        frequencies = np.linspace(freqs[1][0], freqs[1][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_reconstructed_function(
            "Figure 5: Reconstructed signal for frequency range " + str(freqs[1][0]) + " to " + str(freqs[1][1]))

        frequencies = np.linspace(freqs[2][0], freqs[2][1], 1000)
        ft = Fourier(func, func_name, x_values, frequencies)
        ft.plot_reconstructed_function(
            "Figure 6: Reconstructed signal for frequency range " + str(freqs[2][0]) + " to " + str(freqs[2][1]))


if __name__ == "__main__":
    main()

