# this is a test for task 1 with Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT) implementations.


import numpy as np
import matplotlib.pyplot as plt



class Discrete_Fourier_Transform:
    def __init__(self, sampling_rate=100, n=50, wave_velocity=8000, frequency=5):
        self.sampling_rate = sampling_rate
        self.samples = np.arange(n)
        self.wave_velocity = wave_velocity
        self.frequency = frequency
        self.n = n

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

        # Plot signals
        self.plot_signal_with_xy(
            self.samples, signal_A, title="Signal A (Original + Noise)", xlabel="Sample Index", ylabel="Amplitude"
        )
        self.plot_signal_with_xy(
            self.samples, signal_B, title="Signal B (Shifted + Noise)", xlabel="Sample Index", ylabel="Amplitude"
        )

        return signal_A, signal_B

    def fft(self, signal):
        # return np.fft.fft(signal)
        # Recursive implementation of FFT with O(n log n) complexity.
        N = len(signal)

        # Step 1: Pad the signal with zeros if its length is not a power of 2
        if N & (N - 1) != 0:
            next_power_of_2 = 2 ** int(np.ceil(np.log2(N)))
            padded_signal = np.zeros(next_power_of_2, dtype=complex)
            padded_signal[:N] = signal
            signal = padded_signal
            N = len(signal)

        # Step 2: Base case of the recursion
        # like if x[n] = [5] then X[k] = Σ [x[n] * e^(-j*2π*k*n/N)] = 5 where k = 0
        if N <= 1:
            return signal

        # Recursive formula:
        # X[k] = E[k] + e^(-j*2π*k/N) * O[k]

        # Step 3: Divide the signal into even and odd parts and recursively compute FFT on them
        # like even = [signal[0], signal[2], signal[4], ...] and odd = [signal[1], signal[3], signal[5], ...]
        # then recursively compute FFT on even and odd parts
        even = self.fft(signal[::2])
        odd = self.fft(signal[1::2])

        # Step 4: multiply odd part with e^(-j*2π*k/N) and combine the results
        t = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd

        # Step 5: Combine the results
        return np.concatenate([even + t, even - t])


    def ifft(self, signal):
        # Recursive implementation of IFFT
        # Step 1: Take the complex conjugate of the input signal
        conjugated = np.conjugate(signal)

        # Step 2: Compute the FFT of the conjugated signal
        # like FFT(x[n]) = X[k], so FFT(x*[n]) = X*[k], where * is conjugate
        fft_result = self.fft(conjugated)

        # Step 3: Take the complex conjugate of the FFT result
        # like if X[k] = FFT(x[n]), then x[n] = (1 / N) * FFT(X*[k])* where * is conjugate
        result = np.conjugate(fft_result)

        # Step 4: Normalize by dividing by the length of the signal
        # Normalize x[n] = x[n] / N
        return result / len(signal)

    def plot_signal_with_xy(self, x, y, title="Signal", xlabel="X-axis", ylabel="Y-axis", color="blue"):
        plt.figure(figsize=(8, 4))
        plt.stem(x, y, linefmt=color, markerfmt=color, basefmt="gray", label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def cross_correlation(self, signal_A, signal_B):
        # Step 1: Compute DFT of signal A and B
        dft_A = self.fft(signal_A)
        dft_B = self.fft(signal_B)

        # Step 2: Compute conjugate of DFT of signal B
        dft_B_conj = np.conjugate(dft_B)

        # Step 3: Compute cross-correlation in frequency domain
        cross_spectrum = dft_A * dft_B_conj

        # Step 4: Compute inverse DFT to get cross-correlation in time domain
        cross_correlation = self.ifft(cross_spectrum)

        return np.real(cross_correlation)


def main():
    n = 50
    dft = Discrete_Fourier_Transform(n=n)

    noise_freqs = [15, 30, 45]
    amplitudes = [0.5, 0.3, 0.1]
    noise_freqs2 = [10, 20, 40]
    amplitudes2 = [.4, 0.2, 0.2]

    shift_samples = 23
    signal_A, signal_B = dft.generate_signals(noise_freqs, amplitudes, noise_freqs2, amplitudes2, shift_samples)

    # Compute FFT of signals
    spectrum_A = np.abs(dft.fft(signal_A))
    spectrum_B = np.abs(dft.fft(signal_B))

    # Plot spectra
    dft.plot_signal_with_xy(np.arange(len(spectrum_A)), spectrum_A, title="Spectrum of Signal A",
                            xlabel="Frequency Index", ylabel="Magnitude")
    dft.plot_signal_with_xy(np.arange(len(spectrum_B)), spectrum_B, title="Spectrum of Signal B",
                            xlabel="Frequency Index", ylabel="Magnitude")

    # Perform cross-correlation
    cross_relation = dft.cross_correlation(signal_A, signal_B)

    # Plot cross-correlation
    dft.plot_signal_with_xy(np.arange(len(cross_relation)), cross_relation, title="Cross Correlation", xlabel="Lag",
                            ylabel="Correlation", color="green")

    # Find lag
    lag_index = n - np.argmax(cross_relation)
    time_lag = np.abs(lag_index) / dft.sampling_rate
    distance = time_lag * dft.wave_velocity

    print(f"True Shift: {shift_samples} samples")
    print(f"Detected Sample Lag: {lag_index} samples")
    print(f"Time Lag: {time_lag:.4f} seconds")
    print(f"Estimated Distance: {distance:.2f} meters")


if __name__ == "__main__":
    main()