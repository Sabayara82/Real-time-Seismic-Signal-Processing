# ENCM 515 Final Project
# Cython implementation of an earthquake early-warning accelerator simulation
# Group 8


import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class DoubleBuffer:

    #total backing-store length = 2*N
    cdef int capacity 
    #number of samples 
    cdef int num_samples 
    #index of the next write 
    cdef int pointer 
    #how many samples are completed already
    cdef int count 

    cdef double[:] buffer #allows cython to treat this like a C array

    #size int
    #capacity N, the backing store is 2*N based on email exchange w prof
    def __init__(self, int capacity):
        
        self.num_samples = capacity
        self.total_size = 2 * capacity   #double store  
        self.pointer = 0
        self.count = 0
        self.buffer = np.zeros(self.num_samples, dtype=DTYPE)

    #write sample x 
    #into both halves of the backing store
    cpdef void push(self, double x):
        #write to primary position and the mirror position
        self.buffer[self.pointer] = x
        self.buffer[self.pointer + self.num_samples] = x
        #advance pointer and wrap around 
        self.pointer = (self.pointer + 1) % self.num_samples

    cpdef double[:] get_aligned(self):
        return self.buffer[self.pointer : self.pointer + self.num_samples]

# a plain circular buffer that also random access reads with a delay k
#used to hold smoothed energy values for the STA/LTA computation
# with only two subtractions per new sample (O(1))
class EnergyBuffer: 

    cdef int capacity
    cdef int pointer 
    cdef int count 

    cdef double[:] buffer 

    def __init__(self, int capacity):
        self.capacity = capacity
        self.pointer = 0
        self.count = 0

        self.buffer = np.zeros(self.capacity, dtype=DTYPE)

    # override the oldest slot and advance the pointer
    cdef void push(self, double x):
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.capacity

    #return the sample that was pushed k steps ago
    #wraps within the circular store
    cdef double get_delayed(self, int k):
        return self.buffer[(self.pointer - k) % self.capacity]


'''
Band pass filter design
FIR multiplies and past input samples by fixed coefficients and sums them
this is a MAC loop, which is what an accelerator is used for 

the filter is designed in the frequency domain
1. build an ideal rectangular frequency mask
2. inverse-FFT -> time domain impulse response
3. shift so it is casual (zero-phase design)
'''

#f_upper = upper cutoff frequency
#f_lower = lower cutoff frequency
#fs = sampling frequency
#N = filter length (number of taps)

#returns:
# h: ndarray, shape(N,), dtype float64
# the filter's impulse response (coefficients)
cpdef gen_BPF(double f_upper, double f_lower, double fs, int N):
    # allocate the frequency domain mask
    cdef np.ndarray H = np.zeros(N, dtype=complex)

    #
    cdef np.ndarray f_content = np.fft.fftfreq(N, 1.0 / fs)

    cdef int i

    for i in range(N):
        if f_lower <= abs(f_content[i]) <= f_upper:
            H[i] = 1.0
        else:
            H[i] = 0.0

    cdef np.ndarray h = np.fft.ifft(H).real

    h = np.roll(h, N // 2)

    return h


#TODO: accelerator pipeline simulation
