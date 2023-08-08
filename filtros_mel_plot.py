import numpy as np
import matplotlib.pyplot as plt

def mel_filter_bank(num_filters, mel_coefficient, sample_rate, num_fft_points):
    # Definir los límites en la escala Mel
    mel_min = 0
    mel_max = mel_scale(sample_rate / 2, mel_coefficient)

    # Calcular los puntos en la escala Mel para cada filtro triangular
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)

    # Convertir los puntos en Hz utilizando la inversa de la escala Mel
    hz_points = mel_coefficient * (10**(mel_points / 2595) - 1)

    # Calcular los índices en la FFT correspondientes a cada punto en Hz
    fft_indices = np.floor((num_fft_points + 1) * hz_points / sample_rate).astype(int)

    # Crear los filtros triangulares
    filters = np.zeros((num_filters, num_fft_points))
    for i in range(1, num_filters + 1):
        filters[i - 1, fft_indices[i - 1]:fft_indices[i]] = (np.arange(fft_indices[i - 1], fft_indices[i]) - fft_indices[i - 1]) / (fft_indices[i] - fft_indices[i - 1])
        if i < num_filters:
            filters[i - 1, fft_indices[i]:fft_indices[i + 1]] = 1 - (np.arange(fft_indices[i], fft_indices[i + 1]) - fft_indices[i]) / (fft_indices[i + 1] - fft_indices[i])

    return filters

def mel_scale(frequencies, mel_coefficient):
    return 2595 * np.log10(1 + frequencies / mel_coefficient)

num_filters = 40
mel_coefficient = 700
sample_rate = 250
num_fft_points = 128

filters = mel_filter_bank(num_filters, mel_coefficient, sample_rate, num_fft_points)

plt.figure(figsize=(10, 6))
for i in range(num_filters):
    plt.plot(filters[i])

plt.xlabel('Frequency Bin')
plt.ylabel('Filter Amplitude')
plt.title('Mel Filter Bank with Mel Coefficient = 700')
plt.grid(True)
plt.savefig("filtros_mel.png")
plt.show()
