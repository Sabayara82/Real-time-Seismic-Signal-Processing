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
#Main accelerator pipeline simulation loop
#Simulates the hardware stages from raw signal input to earthquake detection trigger
def run_accelerator(np.ndarray[DTYPE_t, ndim=1] s_measured, int NL, int Ns, int Ne, int N, int T,
                    int num_acc_stages, np.ndarray[DTYPE_t, ndim=1] w1,
                    np.ndarray[DTYPE_t, ndim=1] w2, int eq_start, int Max_steps):
        """
        s_measured = pre-padded input signal
        NL = long-term average window
        Ns = short-term average window
        Ne = energy smoothing window
        N = FIR filter length
        T = detection threshold multiplier
        num_acc_stages = pipeline depth
        w1 = BPF FIR coefficients
        w2 = smoothing weights
        eq_start = sample index of injected earthquake
        Max_steps = maximum simulation steps
        """
    #simulation state variables
    cdef int current_step = 0
    cdef int signal_idx = 0
    cdef int stall_count = 0
    cdef int trigger_tripped = 0
    
    #used for internal calculations
    cdef double cur_STA = 0.0
    cdef double cur_LTA = 0.0
    cdef double fir_out_reg
    cdef double energy_sq_reg
    cdef double smooth_en_reg
    cdef int k 

    #memories for high-speed coefficient and signal access
    cdef double[:] fir_weights = w1
    cdef double[:] smooth_weights = w2
    cdef double[:] raw_signal = s_measured

    # Initialize hardware buffer structures
    # raw_input -> fir -> energy -> smoothing -> sta/lta logic
    input_queue = DoubleBuffer(N)
    energy_queue = DoubleBuffer(Ne)
    history_queue = EnergyBuffer(NL) 

    #Pipeline register bank (rows represent stages)
    cdef np.ndarray pipe_regs = np.zeros((num_acc_stages, NL), dtype=DTYPE) 

    #lists for tracking simulation
    lta_history = []
    sta_history = []
    lta_averages = []
    sta_averages = []

    # Main clock-cycle simulation loop
    while (current_step < Max_steps):
        
        #check if the pipeline is still filling the long term window
        if (current_step < NL):
            stall_count += 1

        # STAGE 7: Detection Logic
        # Apply the STA/LTA threshold trigger: (STA/Ns) / (LTA/NL) > T
        if (cur_STA * NL > cur_LTA * T * Ns) and (signal_idx > NL):
            trigger_tripped = 1
            # signal the detection stage and store the triggering values
            pipe_regs[7] = np.ones_like(pipe_regs[7])
            pipe_regs[7][0] = cur_LTA
            pipe_regs[7][1] = cur_STA

        #update for STA/LTA sums to keep O(1) complexity
        #grab the newest smoothed energy value and subtract the delay
        cdef double latest_val = history_queue.buffer[(history_queue.pointer - 1) % history_queue.capacity]
        cur_LTA += latest_val - history_queue.get_delayed(NL)
        cur_STA += latest_val - history_queue.get_delayed(Ns)

        #log status for analysis
        lta_history.append(cur_LTA)
        sta_history.append(cur_STA)            
        lta_averages.append(cur_LTA / NL)
        sta_averages.append(cur_STA / Ns)    

        # STAGE 6: Average Calculation Writeback
        pipe_regs[6][0] = cur_STA
        pipe_regs[6][1] = cur_LTA        

        # STAGE 5: Energy History Buffer
        pipe_regs[5] = np.asarray(history_queue.buffer)

        # STAGE 4: Energy Smoothing FIR
        # Apply weights (w2) to the squared energy values
        smooth_en_reg = 0.0
        for k in range(Ne):
            smooth_en_reg += pipe_regs[3][k] * smooth_weights[k]  
        pipe_regs[4][0] = smooth_en_reg

        # STAGE 3: Energy Window Alignment
        # Fetch contiguous view from the energy double buffer
        cdef double[:] en_window = energy_queue.get_aligned()
        for k in range(Ne):
            pipe_regs[3][k] = en_window[k]      

        # STAGE 2: Energy Transformation (Squaring)
        energy_sq_reg = pipe_regs[1][0] ** 2
        pipe_regs[2][0] = energy_sq_reg

        # STAGE 1: Band-Pass Filter FIR
        # Perform MAC operation between input window and filter coefficients (w1)
        fir_out_reg = 0.0
        cdef double[:] fir_in_view = pipe_regs[0]
        for k in range(N):
            fir_out_reg += fir_in_view[k] * fir_weights[k]  
        pipe_regs[1][0] = fir_out_reg

        # STAGE 0: Data Acquisition / Alignment
        # Align the raw circular buffer into the first pipeline stage
        cdef double[:] raw_window = input_queue.get_aligned()
        for k in range(N):
            pipe_regs[0][k] = raw_window[k]            

        # Update buffers for the next clock cycle
        current_step += 1
        input_queue.push(raw_signal[signal_idx])
        energy_queue.push(pipe_regs[2][0])
        history_queue.push(pipe_regs[4][0])
        
        signal_idx += 1

        # Terminate simulation if detection occurs
        if (trigger_tripped == 1):
            break           

    # Final simulation report
    return {
        'total_steps' : current_step,
        'stalled_cycles' : stall_count,
        'is_detected' : trigger_tripped == 1,
        'lta_history' : lta_history,
        'sta_history' : sta_history,
        'lta_averages' : lta_averages,
        'sta_averages' : sta_averages,
        'eq_start_idx' : eq_start
    }
