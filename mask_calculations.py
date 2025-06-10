# masking calculations for 1D FFTs
import numpy as np


def calculate_W(weights):
    '''
    calculates the average FFT of the weights.
    weights (np.ndarray): Weights in real-pixel space.
        N_q x N array, where N_q is the number of quasars and N is the length of the pixel grid
    Returns:
    W (np.ndarray): Average of the Fourier-transformed weights-magnitude-squared, vector of length N
    '''
    w = np.fft.fft(weights, axis=1)
    return np.sum((w.real**2 + w.imag**2), axis=0)/w.shape[0]
    
    
def calculate_window_matrix(weights, resolution, L):
    '''
    weights (np.ndarray): array N_q [number of quasar spectra] x N [Number of FFT pixels],
         the real-space pixel weights for each skewer
    resolution (np.ndarray): Fourier-space resolution function evaluated at pixel coordinates; vector length N
    L (float): length of skewers

    Returns:
    window_matrix (np.ndarray): NxN window matrix
    estnorm (np.ndarray): vector length N, normalization for the estimated P1D (after averaging over N_q quasars)
    W (np.ndarray): vector length N, average FFT of the weights 
    '''
    W = calculate_W(weights)
    R2 = resolution.real**2 + resolution.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
    estnorm = np.absolute(L/denom)
    N = estnorm.size
    window_matrix = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            window_matrix[m,n] = W[m-n]*R2[n] / denom[m]
    return window_matrix, estnorm, W

def masked_theory(window_matrix, model):
    '''
    Calculate the prediction for weighted P1D theory.
    window_matrix (np.ndarray): Real-valued NxN window matrix, where N is FFT grid length
    model (np.ndarray): Real-valued vector length N of the original theory model
    '''
    return np.matmul(window_matrix, model)
    