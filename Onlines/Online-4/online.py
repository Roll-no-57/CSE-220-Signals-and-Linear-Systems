import numpy as np
import matplotlib.pyplot as plt


class Discrete_Fourier_Transform:
    def __init__(self):
        pass

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

        x_range = np.linspace(0, len(cross_correlation) - 1, len(cross_correlation))
        x_range[x_range > len(x_range) / 2] -= len(x_range)
        x_range = -x_range

        return np.real(cross_correlation), x_range

    def sample_lag_detection(cross_correlation, x_range):
        return int(x_range[np.argmax(np.abs(cross_correlation))])





# Example usage
x = 65767879797907
y = 765454532435435345

#converting to digit arrays(discrete signal)
x_digits = [int(digit) for digit in str(x)]
# print(x_digits)

y_digits = [int(digit) for digit in str(y)]
# print(y_digits)



n1 = len(x_digits)
n2 = len(y_digits)

pad = n1+n2-1
next_pow = 2 ** int(np.ceil(np.log2(pad)))

padded_n1 = np.zeros(next_pow)
padded_n1[:n1] = x_digits
x_digits = padded_n1
n1 = len(x_digits)

padded_n2 = np.zeros(next_pow)
padded_n2[:n2] = y_digits
y_digits = padded_n2
n2 = len(y_digits)

# print(n2,n1)
# print(padded_n2,padded_n1)


dft = Discrete_Fourier_Transform()

n1_fft = dft.fft(padded_n1)
n2_fft = dft.fft(padded_n2)

# print(n1_fft)
# print(n2_fft)

fft_mull = n1_fft * n2_fft

product_digits = dft.ifft(fft_mull).real[:pad]
# print(product_digits)
product_digits = np.round(product_digits).astype(int)


carry = 0
for i in range(len(product_digits) - 1, -1, -1):
    product_digits[i] += carry
    carry = product_digits[i] // 10
    product_digits[i] %= 10

# Handle any remaining carry
result_digits = list(product_digits)
while carry > 0:
    result_digits.insert(0, carry % 10)
    carry //= 10

# Convert digits to a continuous number string
result_str = ''.join(str(d) for d in result_digits)
print("Result:", result_str)

