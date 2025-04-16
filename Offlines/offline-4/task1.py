import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


class Discrete_Fourier_Transform:

    def __init__(self, sampling_rate=100, n=50, wave_velocity=8000, frequency=5):
        self.sampling_rate = sampling_rate
        self.samples = np.arange(n)
        self.wave_velocity = wave_velocity
        self.frequency = frequency
        self.n = n

    def dft(self, signal):
        n = self.n
        dft_real = np.zeros(n)
        dft_imag = np.zeros(n)

        # Compute DFT
        # Discrete Fourier Transform (DFT):
        # X(k) = ∑ x[n] * e^(-j*2π*k*n/N)
        # X(k) = Σ [x(n) * (cos(2π * k * n / N) - j * sin(2π * k * n / N))]
        # where Σ is the summation from n=0 to N-1.

        for k in range(n):
            for t in range(n):
                dft_real[k] += signal[t] * np.cos(2 * np.pi * k * t / n)
                dft_imag[k] += -1 * signal[t] * np.sin(2 * np.pi * k * t / n)

        return dft_real, dft_imag

    def inverse_dft(self, dft_real, dft_imag):
        n = self.n
        reconstructed_signal = np.zeros(n)

        # Compute Inverse DFT
        # Inverse Discrete Fourier Transform (IDFT):
        # x(n) = (1 / N) * Σ [X(k)_real * (cos(2π * k * n / N) - j * X(k)_imag * sin(2π * k * n / N))]
        # where Σ is the summation from k=0 to N-1.
        for t in range(n):
            for k in range(n):
                reconstructed_signal[t] += (1 / n) * (
                    dft_real[k] * np.cos(2 * np.pi * k * t / n)
                    - dft_imag[k] * np.sin(2 * np.pi * k * t / n)
                )

        return reconstructed_signal

    def cross_correlation(self, signal_A, signal_B):

        # Step 1: Compute DFT of signal A and B
        dft_signal_A_real, dft_signal_A_imag = self.dft(signal_A)  # x1[n] <--DFT--> X1[k]
        dft_signal_B_real, dft_signal_B_imag = self.dft(signal_B)  # x2[n] <--DFT--> X2[k]

        # Step 2: Compute conjugate of DFT of signal B
        dft_signal_B_conj_real = dft_signal_B_real
        dft_signal_B_conj_imag = -1 * dft_signal_B_imag # X2[k] <--Conjugate DFT--> X2*[k]

        # Step 3: Compute cross-correlation in frequency domain
        # X[k] ----> X1[k] * X2*[k]
        dft_transform_real = (
            dft_signal_A_real * dft_signal_B_conj_real
            - dft_signal_A_imag * dft_signal_B_conj_imag
        )
        dft_transform_imag = (
            dft_signal_A_real * dft_signal_B_conj_imag
            + dft_signal_A_imag * dft_signal_B_conj_real
        )

        # Step 4: Compute inverse DFT to get cross-correlation in time domain
        # X[k] ----> x[n]
        cross_correlation = self.inverse_dft(dft_transform_real, dft_transform_imag)

        # x_range = np.linspace(0, len(cross_correlation) - 1, len(cross_correlation))
        # x_range[x_range > len(x_range) / 2] -= len(x_range)
        # x_range = -x_range

        return cross_correlation


    def generate_signals(self, noise_freqs_A, noise_amplitudes_A, noise_freqs_B, noise_amplitudes_B, shift_samples):
        dt = 1 / self.sampling_rate
        time = self.samples * dt

        # Original clean signal
        original_signal = np.sin(2 * np.pi * self.frequency * time)

        # Add noise to Signal A
        noise_for_signal_A = sum(
            amplitude * np.sin(2 * np.pi * noise_freq * time)
            for noise_freq, amplitude in zip(noise_freqs_A, noise_amplitudes_A)
        )
        signal_A = original_signal + noise_for_signal_A

        # Add noise to Signal B
        noise_for_signal_B = sum(
            amplitude * np.sin(2 * np.pi * noise_freq * time)
            for noise_freq, amplitude in zip(noise_freqs_B, noise_amplitudes_B)
        )
        noisy_signal_B = signal_A + noise_for_signal_B

        # Shift Signal B
        signal_B = np.roll(noisy_signal_B, shift_samples)
        # signal_B = noisy_signal_B

        # Plot signals
        self.plot_signal_with_xy(
            self.samples, signal_A, title="Signal A (Original + Noise)", xlabel="Sample Index", ylabel="Amplitude"
        )
        self.plot_signal_with_xy(
            self.samples, signal_B, title="Signal B (Shifted + Noise)", xlabel="Sample Index", ylabel="Amplitude"
        )

        return signal_A, signal_B

    def plot_signal_with_xy(self, x, y, title="Signal", xlabel="X-axis", ylabel="Y-axis", color="blue"):
        plt.figure(figsize=(8, 4))
        plt.stem(x, y, linefmt=color, markerfmt=color, basefmt="gray", label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def plot_cross_correlation(self, correlation, title="Cross Correlation", xlabel="Lag (samples)",
                               ylabel="Correlation", color="green"):
        half_range = len(correlation) // 2
        lags = np.arange(-half_range, half_range)
        plt.figure(figsize=(8, 4))
        plt.stem(lags, correlation, linefmt=color, markerfmt=color, basefmt="gray", label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def low_pass_filter(self, signal, cutoff, order=5):
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal


def main():
    n = 50
    # Initialize DFT object
    dft = Discrete_Fourier_Transform(n=n)

    # test case sir provided
    # noise_freqs = [15, 30, 45]
    # amplitudes = [0.5, 0.3, 0.1]
    # noise_freqs2 = [10, 20, 40]
    # amplitudes2 = [0.3, 0.2, 0.1]

    noise_freqs = [15, 30, 45]
    amplitudes = [0.5, 0.3, 0.1]
    noise_freqs2 = [10, 20, 40]
    amplitudes2 = [.4, 0.2, 0.2]


    # Shift signal B
    # shift_samples = np.random.randint(-self.n // 2, self.n // 2)
    shift_samples = 23  # sir test case
    # Generate signals
    signal_A, signal_B = dft.generate_signals(noise_freqs, amplitudes, noise_freqs2, amplitudes2, shift_samples)

    # Compute DFT of Signal A and B
    dft_A_real, dft_A_imag = dft.dft(signal_A)
    dft_B_real, dft_B_imag = dft.dft(signal_B)

    # Compute magnitude spectra
    spectrum_A = np.sqrt(dft_A_real**2 + dft_A_imag**2)
    spectrum_B = np.sqrt(dft_B_real**2 + dft_B_imag**2)

    # Plot spectra
    dft.plot_signal_with_xy(dft.samples, spectrum_A, title="Spectrum of Signal A", xlabel="Frequency Index", ylabel="Amplitude")
    dft.plot_signal_with_xy(dft.samples, spectrum_B, title="Spectrum of Signal B", xlabel="Frequency Index", ylabel="Amplitude")

    # Perform cross-correlation
    cross_relation = dft.cross_correlation(signal_A, signal_B)

    # Find lag and calculate distance
    lag_index = n-np.argmax(np.abs(cross_relation))
    time_lag = np.abs(lag_index) / dft.sampling_rate
    distance = time_lag * dft.wave_velocity


    # Plot cross-correlation with centered lags
    dft.plot_cross_correlation(
        cross_relation,
        title="Cross Correlation of Signal A and Signal B",
        xlabel="Lag (samples)",
        ylabel="Correlation",
        color="green",
    )

    # Print results
    print(f"True Shift: {shift_samples} samples")
    print(f"Detected Sample Lag: {lag_index} samples")
    print(f"Time Lag: {time_lag:.4f} seconds")
    print(f"Estimated Distance: {distance:.2f} meters")

    ######################################-------FILTERING-------######################################

    # Apply low-pass filter to reduce noise
    cutoff_frequency = 6  # Adjust cutoff frequency for filtering # use 15
    filtered_signal_A = dft.low_pass_filter(signal_A, cutoff=cutoff_frequency)
    filtered_signal_B = dft.low_pass_filter(signal_B, cutoff=cutoff_frequency)
    # use second filter 5 to show the difference

    # Plot filtered signals
    dft.plot_signal_with_xy(
        dft.samples, filtered_signal_A, title="Filtered Signal A", xlabel="Sample Index", ylabel="Amplitude"
    )
    dft.plot_signal_with_xy(
        dft.samples, filtered_signal_B, title="Filtered Signal B", xlabel="Sample Index", ylabel="Amplitude"
    )

    # Compute DFT of Signal A and B
    dft_A_filtered_real, dft_A_filtered_imag = dft.dft(filtered_signal_A)
    dft_B_filtered_real, dft_B_filtered_imag = dft.dft(filtered_signal_B)

    # Compute magnitude spectra
    spectrum_A_filtered = np.sqrt(dft_A_filtered_real**2 + dft_A_filtered_imag**2)
    spectrum_B_filtered = np.sqrt(dft_B_filtered_real**2 + dft_B_filtered_imag**2)

    # Plot spectra
    dft.plot_signal_with_xy(dft.samples, spectrum_A_filtered, title="Spectrum of Signal A with filtered", xlabel="Frequency Index", ylabel="Amplitude")
    dft.plot_signal_with_xy(dft.samples, spectrum_B_filtered, title="Spectrum of Signal B with filtered", xlabel="Frequency Index", ylabel="Amplitude")

    # Perform cross-correlation on filtered signals
    cross_relation_filtered = dft.cross_correlation(filtered_signal_A, filtered_signal_B)
    # Plot cross-correlation filtered results

    # Plot cross-correlation with centered lags
    dft.plot_cross_correlation(
        cross_relation_filtered,
        title="Cross Correlation of Signal A and Signal B",
        xlabel="Lag (samples)",
        ylabel="Correlation",
        color="green",
    )

    # Find lag and calculate distance
    lag_index_filter = n-np.argmax(cross_relation_filtered)
    time_lag_filtered = np.abs(lag_index_filter) / dft.sampling_rate
    distance_filetered = time_lag_filtered * dft.wave_velocity

    # Print results
    print(f"True Shift: {shift_samples} samples")
    print(f"Detected Sample Lag filtered: {lag_index_filter} samples")
    print(f"Time Lag filtered: {time_lag_filtered:.4f} seconds")
    print(f"Estimated Distance filtered: {distance_filetered:.2f} meters")


# Run the main function
if __name__ == "__main__":
    main()
    # done ♥
