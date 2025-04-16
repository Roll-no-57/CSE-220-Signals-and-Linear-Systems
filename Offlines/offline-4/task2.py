
import numpy as np
import time
import matplotlib.pyplot as plt


class FourierTransform:
    def __init__(self):
        pass

    def dft(self, signal):
        n = len(signal)
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
        n = len(dft_real)
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

    def run_experiment(self):
        sizes = [2 ** k for k in range(2, 11)]  # n = 4, 8, 16, ..., 1024
        dft_times = []
        fft_times = []
        idft_times = []
        ifft_times = []
        for n in sizes:
            signal = np.random.rand(n)

            # Measure DFT runtime
            start = time.time()
            dft_real, dft_imag = self.dft(signal)
            dft_times.append(time.time() - start)

            # Measure FFT runtime
            start = time.time()
            fft_result = self.fft(signal)
            fft_times.append(time.time() - start)

            # Measure IDFT runtime
            start = time.time()
            self.inverse_dft(dft_real, dft_imag)
            idft_times.append(time.time() - start)

            # Measure IFFT runtime
            start = time.time()
            self.ifft(fft_result)
            ifft_times.append(time.time() - start)

        return sizes, dft_times, fft_times, idft_times, ifft_times

    def plot_results(sizes, dft_times, fft_times, idft_times, ifft_times, filename="fft_vs_dft_plot.png"):
        plt.figure(figsize=(12, 6))

        # Plot DFT and FFT runtimes
        plt.subplot(1, 2, 1)
        plt.plot(sizes, dft_times, label="DFT", marker="o")
        plt.plot(sizes, fft_times, label="FFT", marker="o")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Signal Size (n)")
        plt.ylabel("Runtime (seconds)")
        plt.title("DFT vs FFT Runtime")
        plt.legend()
        plt.grid()

        # Plot IDFT and IFFT runtimes
        plt.subplot(1, 2, 2)
        plt.plot(sizes, idft_times, label="IDFT", marker="o")
        plt.plot(sizes, ifft_times, label="IFFT", marker="o")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Signal Size (n)")
        plt.ylabel("Runtime (seconds)")
        plt.title("IDFT vs IFFT Runtime")
        plt.legend()
        plt.grid()

        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.show()


def main():
    # Run the experiment
    ft = FourierTransform()
    sizes, dft_times, fft_times, idft_times, ifft_times = ft.run_experiment()

    # Plot results
    FourierTransform.plot_results(sizes, dft_times, fft_times, idft_times, ifft_times)


if __name__ == "__main__":
    main()
