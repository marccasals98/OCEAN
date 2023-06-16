# implementation inspired by https://github.com/tracek/Ornithokrites/blob/master/noise_subtraction.py

import scipy as sp
from scipy.fft import fft, ifft
import scipy.signal as ssig


class SpectralSubtraction():
    def __init__(self, winsize, window, coefficient=5.0, ratio=1.0):
        self._window = window
        self._coefficient = coefficient
        self._ratio = ratio

    def compute(self, signal, noise):
        n_spec = sp.fft(noise*self._window)
        n_pow = sp.absolute(n_spec)**2.0
        return self.compute_by_noise_pow(signal, n_pow)

    def compute_by_noise_pow(self, signal, n_pow, add_noisy_phase=True):
        #print(sp, type(sp))
        #print(sp.fft, type(sp.fft))
        s_spec = fft(signal*self._window)
        s_amp = sp.absolute(s_spec)
        s_phase = sp.angle(s_spec)
        s_amp2 = s_amp**2.0
        amp = s_amp2 - n_pow*self._coefficient
        # amp = s_amp**2.0 - (1 + np.std(n_pow) / n_pow) * n_pow * 2

        amp = sp.maximum(amp, 0.01 * s_amp2)
        amp = sp.sqrt(amp)
        amp = self._ratio*amp + (1.0-self._ratio)*s_amp
        if add_noisy_phase:
          spec = amp * sp.exp(s_phase*1j)
        else:
          spec = amp
        return sp.real(ifft(spec))


class SpectrumReconstruction(object):
    def __init__(self, winsize, window, constant=0.001, ratio=1.0, alpha=0.99):
        self._window = window
        self._G = sp.zeros(winsize, sp.float32)
        self._prevGamma = sp.zeros(winsize, sp.float32)
        self._alpha = alpha
        self._prevAmp = sp.zeros(winsize, sp.float32)
        self._ratio = ratio
        self._constant = constant

    def compute(self, signal, noise):
        n_spec = sp.fft(noise*self._window)
        n_pow = sp.absolute(n_spec)**2.0
        return self.compute_by_noise_pow(signal, n_pow)

    def _calc_aposteriori_snr(self, s_amp, n_pow):
        return s_amp**2.0/n_pow

    def _calc_apriori_snr(self, gamma):
        return self._alpha*self._G**2.0 * self._prevGamma + (1.0 - self._alpha) * sp.maximum(gamma - 1.0, 0.0)

    def _calc_apriori_snr2(self, gamma, n_pow):
        return self._alpha*(self._prevAmp**2.0/n_pow) + (1.0-self._alpha)*sp.maximum(gamma-1.0, 0.0)


def get_frame(signal, winsize, no):
    shift = winsize / 2
    #shift = winsize / 1.5
    start = no * shift
    end = start+winsize
    #return signal[start:end]
    return signal[int(start):int(end)]


def add_signal(signal, frame, winsize, no):
    shift = winsize / 2
    #shift = winsize / 1.5
    start = no * shift
    end = start + winsize
    start = int(start) # added by me
    end = int(end) # added by me
    signal[start:end] = signal[start:end] + frame


def reduce_noise(signal, noisy_signal, winsize=2**10, add_noisy_phase=True):
    """ Reduce noise """
    window=sp.hanning(winsize)
    method = SpectralSubtraction(winsize, window)

    out = sp.zeros(len(signal), sp.float32)
    
    power = ssig.welch(noisy_signal, window=window, return_onesided=False, scaling='spectrum')[1] * window.sum()**2
    nf = len(signal)/(winsize/2) - 1
    #nf = len(signal)/(winsize/1.5) - 1
    for no in range(int(nf)):
        s = get_frame(signal, winsize, no)
        add_signal(out, method.compute_by_noise_pow(s, power, add_noisy_phase), winsize, no)
    return out


def get_noise(signal, rate, intervals):
    interval = intervals.popitem()
    if interval[1][0] == 0:
        start = 0
    else:
        start = interval[1][0] + 3*rate
    end = interval[1][1] - rate
    return signal[start:end]