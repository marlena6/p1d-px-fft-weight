# masking calculations for 1D FFTs
import numpy as np
import numpy.fft as fft
from numba import jit




def calculate_mask_mn_rfft(mask_array_rfft, m, n, N, Nq):
    # calculate the mask matrix
    # mask_array_fft is the rFFT of the mask array. Nq x Nk
    # m is the index of k bins we want to compute

    l = m-n
    sum_over_quasars = 0

    if (l >= 0) & (l<=N//2):
        w_lq = mask_array_rfft[:, l]
    elif (l >= 0) & (l>N//2):
        w_lq = np.conjugate(mask_array_rfft[:, [l-(N//2)]])
    elif (l < 0)&(np.abs(l)<=N//2):
        w_lq = np.conjugate(mask_array_rfft[:, np.abs(l)])
    elif (l < 0)&(np.abs(l)>N//2):
        w_lq = mask_array_rfft[:, [abs(l)-(N//2)]]
    
    sum_over_quasars = np.sum(np.abs(w_lq)**2)
    return sum_over_quasars/Nq/N**2

@jit
def calculate_mask_mn_fft_loop(mask_array_fft, m, n, N, Nq):
    # calculate the mask matrix
    # mask_array_fft is the FFT of the mask array. Nq x Nk
    # m is the index of k bins we want to compute
    l = m-n
    if l>=0:
        w_lq = mask_array_fft[:, l]
    else:
        w_lq = mask_array_fft[:, N-np.abs(l)]
    sum_over_quasars = np.sum(np.abs(w_lq)**2)
    return sum_over_quasars/Nq/N**2

@jit
def calculate_mask_mn_fft_array(mask_array_fft, m, n, N, Nq):
    print("doing the array mode.")
    l = m-n
    w_lq = np.zeros_like(mask_array_fft)
    l_lt0 = l<0
    l_gt0 = l>=0
    w_lq[:,l_lt0] = mask_array_fft[:, N-np.abs(l[l_lt0])]
    w_lq[:,l_gt0] = mask_array_fft[:, l[l_gt0]]
    sum_over_quasars = np.sum(np.abs(w_lq)**2, axis=0)
    return sum_over_quasars/Nq/N**2

def calculate_masked_power_rfft(m, mask_array_rfft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_rfft.shape[0]
    N = mask_array_rfft.shape[1]*2-1
    masked_power = 0
    for n in range(N//2+1):
        masked_power += theory_power[n] * calculate_mask_mn_rfft(mask_array_rfft, m, n, N, Nq)
        # print("n=",n, "m-n=",m-n, "getting this element of the theory power", n)
    return masked_power
        
    # return matrix of n x q, where q is quasar index and n is number of pixels

@jit
def calculate_masked_power_fft_loop(m, mask_array_fft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_fft.shape[0]
    N = mask_array_fft.shape[1]
    n = np.arange(N)
    masked_power = 0
    for n in range(N):
        masked_power += theory_power[n] * calculate_mask_mn_fft_loop(mask_array_fft, m, n, N, Nq)
    return masked_power

@jit
def calculate_masked_power_fft_array(m, mask_array_fft, theory_power):
    # theory_power is a length Np vector
    Nq = mask_array_fft.shape[0]
    N = mask_array_fft.shape[1]
    n = np.arange(N)
    masked_power = 0
    mask_mn_fft = calculate_mask_mn_fft_array(mask_array_fft, m, n, N, Nq)
    print("got the mask_mn_fft array")
    for n in range(N):
        masked_power += theory_power[n] * mask_mn_fft[n]

def calculate_masked_power(avg_sqmask_array_fft, theory_power):
    avg_sqmask_array_2fft = np.fft.fft(avg_sqmask_array_fft)
    del avg_sqmask_array_fft
    theory_power_2fft = np.fft.fft(theory_power)
    del theory_power
    masked_power_fft = avg_sqmask_array_2fft * theory_power_2fft
    del avg_sqmask_array_2fft
    del theory_power_2fft
    masked_power = np.fft.ifft(masked_power_fft)
    return masked_power

# modify all skewers such that they begin at random locations along the line-of-sight, and wrap around. All should be 1/2 length of box.
def randomize_skewer_start(skewer_grid, rng):
    nside = skewer_grid.shape[0]
    Np = skewer_grid.shape[2]
    new_skewer_length = Np//2
    new_skewer_grid = np.zeros((nside, nside, new_skewer_length))
    for i in range(nside):
        for j in range(nside):
            start = rng.integers(Np)
            if start+new_skewer_length > Np:
                remainder = new_skewer_length - (Np-start)
                new_skewer_grid[i,j,:] = np.concatenate((skewer_grid[i,j,start:], skewer_grid[i,j,:remainder]))
            else:
                new_skewer_grid[i,j,:] = skewer_grid[i,j,start:start+new_skewer_length]
    return new_skewer_grid, new_skewer_length

# modify all skewers such that they are all only 1/2 length of box, start at random locations, and have zero padding elsewhere
def skewer_offset_and_pad(skewer_grid, rng, return_mask=True, randomize_skewer_lengths=False):
    mask = np.ones(skewer_grid.shape)
    new_skewer_grid = np.ma.MaskedArray.copy(skewer_grid)
    nside = skewer_grid.shape[0]
    Np = skewer_grid.shape[2]
    if not randomize_skewer_lengths:
        pad_length = Np//3
    for i in range(nside):
        for j in range(nside):
            pad_start = rng.integers(Np)
            if randomize_skewer_lengths:
                pad_length = rng.integers(int(Np//4), int(Np//1.5))
            if pad_start + pad_length > Np:
                remainder = pad_length - (Np-pad_start)
                # new_skewer_grid[i,j,pad_start:] = 0
                # new_skewer_grid[i,j,:remainder] = 0
                mask[i,j,pad_start:] = 0
                mask[i,j,:remainder] = 0
            else:
                # new_skewer_grid[i,j,pad_start:pad_start+pad_length] = 0
                mask[i,j,pad_start:pad_start+pad_length] = 0
    mask = np.asarray(mask)
    if return_mask:
        return new_skewer_grid*mask, mask
    else:
        return new_skewer_grid*mask
    
def calculate_W(weights):
    '''
    calculates the average FFT of the weights.
    weights (np.ndarray): N_q x N array, where N_q is the number of quasars and N is the length of the pixel grid
    Returns:
    W (np.ndarray): vector of length N
    '''
    w = np.fft.fft(weights, axis=1)
    return np.sum((w.real**2 + w.imag**2), axis=0)/w.shape[0]
    
    
def calculate_window_matrix(weights, resolution, L):
    '''
    resolution (np.ndarray): vector length N
    weights (np.ndarray): array N_q x N, the real-space pixel weights for each skewer
    L (float): length of skewers
    Returns:
    window_matrix (np.ndarray): Real-valued NxN window matrix
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
    