import numpy as np
import time
import matplotlib.pyplot as plt


def generate_random_signal(size):
    return np.random.rand(size)


def dft(signal):
    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return dft_result


def idft(frequency_signal):
    N = len(frequency_signal)
    time_signal = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            time_signal[n] += frequency_signal[k] * np.exp(2j * np.pi * k * n / N)
    return time_signal / N


def fft(signal):
    N = len(signal)
    if N & (N - 1) != 0:
        next_power_of_2 = 2 ** int(np.ceil(np.log2(N)))
        padded_signal = np.zeros(next_power_of_2, dtype=complex)
        padded_signal[:N] = signal
        signal = padded_signal
        N = len(signal)
    if N <= 1:
        return signal
    even = fft(signal[::2])
    odd = fft(signal[1::2])
    t = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + t, even - t])


def ifft(signal):
    N = len(signal)
    if N & (N - 1) != 0:
        next_power_of_2 = 2 ** int(np.ceil(np.log2(N)))
        padded_signal = np.zeros(next_power_of_2, dtype=complex)
        padded_signal[:N] = signal
        signal = padded_signal
        N = len(signal)
    if N <= 1:
        return signal
    even = ifft(signal[::2])
    odd = ifft(signal[1::2])
    t = np.exp(2j * np.pi * np.arange(N // 2) / N) * odd
    result = np.concatenate([even + t, even - t]) / 2
    if len(result) > N:
        result = result[:N]
    return result


def measure_runtime(func, signal, iterations=10):
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(signal)
        times.append(time.perf_counter() - start_time)
    return np.mean(times)


# Plot runtime comparisons
def plot_runtime(signal_sizes, runtimes_dft, runtimes_fft, title, ylabel, label1, label2):
    plt.figure(figsize=(10, 6))
    plt.plot(signal_sizes, runtimes_dft, label=label1, marker='o', color='blue')
    plt.plot(signal_sizes, runtimes_fft, label=label2, marker='o', color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Signal Size (n)')
    plt.ylabel(ylabel)
    plt.title(title)
    log_n_values = np.log2(signal_sizes)
    plt.xticks(signal_sizes, [f"$2^{{{int(log_n)}}}$" for log_n in log_n_values])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_signals(original_signal, dft_result, reconstructed_signal):
    N = len(original_signal)
    freq = np.arange(N)  # Frequency indices for the DFT result

    plt.figure(figsize=(12, 8))

    # Plot Original Signal
    plt.subplot(3, 1, 1)
    plt.plot(original_signal, label="Original Signal", color="blue", marker='o')
    plt.title("Original Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot DFT Magnitude Spectrum
    plt.subplot(3, 1, 2)
    plt.stem(freq, np.abs(dft_result), label="DFT Magnitude Spectrum", linefmt="orange", markerfmt="C1o", basefmt="C1-")
    plt.title("DFT Magnitude Spectrum")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()

    # Plot Reconstructed Signal
    plt.subplot(3, 1, 3)
    plt.plot(reconstructed_signal.real, label="Reconstructed Signal (Real Part)", color="green", linestyle="--",
             marker='x')
    plt.title("Reconstructed Signal After IDFT")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main Execution
def main():
    signal_sizes = [2 ** k for k in range(2, 12)]
    dft_times, fft_times = [], []
    idft_times, ifft_times = [], []

    for n in signal_sizes:
        signal = generate_random_signal(n)

        dft_times.append(measure_runtime(dft, signal))
        fft_times.append(measure_runtime(fft, signal))

        frequency_signal = fft(signal)
        idft_times.append(measure_runtime(idft, frequency_signal))
        ifft_times.append(measure_runtime(ifft, frequency_signal))

    dft_result = dft(signal)

    # Apply IDFT
    reconstructed_signal = idft(dft_result)

    # Plot the signals
    plot_signals(signal, dft_result, reconstructed_signal)
    plot_runtime(signal_sizes, dft_times, fft_times, "DFT vs FFT Runtime", "Runtime (seconds)", "DFT Runtime",
                 "FFT Runtime")
    plot_runtime(signal_sizes, idft_times, ifft_times, "IDFT vs IFFT Runtime", "Runtime (seconds)", "IDFT Runtime",
                 "IFFT Runtime")


if __name__ == "__main__":
    main()
