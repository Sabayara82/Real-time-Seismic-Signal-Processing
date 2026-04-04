# ENCM 515 Final Project
# Cython implementation of an earthquake early-warning accelerator simulation
# Group 8


import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# backing store is 2*N based on email exchange w prof
# lets get_aligned always return a contiguous slice without copying
cdef class DoubleBuffer:

    #number of samples 
    cdef int num_samples 
    #index of the next write 
    cdef int pointer 
    cdef int count 

    cdef double[:] buffer

    def __init__(self, int capacity):
        self.num_samples = capacity
        self.pointer = 0
        self.count = 0
        # FIX: allocate 2*num_samples (was only num_samples, push would go OOB)
        self.buffer = np.zeros(2 * self.num_samples, dtype=DTYPE)

    #write sample x into both halves
    cpdef void push(self, double x):
        self.buffer[self.pointer] = x
        self.buffer[self.pointer + self.num_samples] = x
        self.pointer = (self.pointer + 1) % self.num_samples

    cpdef double[:] get_aligned(self):
        return self.buffer[self.pointer : self.pointer + self.num_samples]


# plain circular buffer w/ delayed random access
# O(1) STA/LTA - just subtract the sample falling out of the window
cdef class EnergyBuffer: 

    cdef int capacity
    cdef int pointer 
    cdef int count 
    cdef double[:] buffer 

    def __init__(self, int capacity):
        self.capacity = capacity
        self.pointer = 0
        self.count = 0
        self.buffer = np.zeros(self.capacity, dtype=DTYPE)

    # FIX: was cdef (C-only, invisible to pipeline), needs to be cpdef
    cpdef void push(self, double x):
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.capacity

    cpdef double get_delayed(self, int k):
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
    # FIX: all cdef declarations must come before any executable statements in Cython
    cdef np.ndarray H = np.zeros(N, dtype=complex)
    cdef np.ndarray f_content = np.fft.fftfreq(N, 1.0 / fs)
    cdef np.ndarray h
    cdef int i

    for i in range(N):
        if f_lower <= abs(f_content[i]) <= f_upper:
            H[i] = 1.0
        else:
            H[i] = 0.0

    h = np.fft.ifft(H).real
    h = np.roll(h, N // 2)

    return h


#TODO: accelerator pipeline simulation
#Main accelerator pipeline simulation loop
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
    # FIX: all cdef declarations moved to top of function (Cython requires this)
    cdef int current_step = 0
    cdef int signal_idx = 0
    cdef int stall_count = 0
    cdef int trigger_tripped = 0
    
    cdef double cur_STA = 0.0
    cdef double cur_LTA = 0.0
    cdef double fir_out_reg
    cdef double energy_sq_reg
    cdef double smooth_en_reg
    cdef double latest_val
    cdef int k 

    cdef double[:] fir_weights = w1
    cdef double[:] smooth_weights = w2
    cdef double[:] raw_signal = s_measured
    cdef double[:] en_window
    cdef double[:] fir_in_view
    cdef double[:] raw_window

    # Initialize hardware buffer structures
    # raw_input -> fir -> energy -> smoothing -> sta/lta logic
    input_queue = DoubleBuffer(N)
    energy_queue = DoubleBuffer(Ne)
    history_queue = EnergyBuffer(NL) 

    #Pipeline register bank (rows represent stages)
    cdef np.ndarray pipe_regs = np.zeros((num_acc_stages, NL), dtype=DTYPE) 

    lta_history = []
    sta_history = []
    lta_averages = []
    sta_averages = []

    while (current_step < Max_steps):
        
        # stall until long-term window is full
        if (current_step < NL):
            stall_count += 1

        # STAGE 7: Detection Logic
        # cross multiply to avoid division: STA/Ns > T * LTA/NL
        if (cur_STA * NL > cur_LTA * T * Ns) and (signal_idx > NL):
            trigger_tripped = 1
            pipe_regs[7] = np.ones_like(pipe_regs[7])
            pipe_regs[7][0] = cur_LTA
            pipe_regs[7][1] = cur_STA

        # O(1) running sum - add newest sample, drop the one falling out
        latest_val = history_queue.buffer[(history_queue.pointer - 1) % history_queue.capacity]
        cur_LTA += latest_val - history_queue.get_delayed(NL)
        cur_STA += latest_val - history_queue.get_delayed(Ns)

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
        smooth_en_reg = 0.0
        for k in range(Ne):
            smooth_en_reg += pipe_regs[3][k] * smooth_weights[k]  
        pipe_regs[4][0] = smooth_en_reg

        # STAGE 3: Energy Window Alignment
        en_window = energy_queue.get_aligned()
        for k in range(Ne):
            pipe_regs[3][k] = en_window[k]      

        # STAGE 2: Energy Transformation (Squaring)
        energy_sq_reg = pipe_regs[1][0] ** 2
        pipe_regs[2][0] = energy_sq_reg

        # STAGE 1: Band-Pass Filter FIR - MAC loop (hot path)
        fir_out_reg = 0.0
        fir_in_view = pipe_regs[0]
        for k in range(N):
            fir_out_reg += fir_in_view[k] * fir_weights[k]  
        pipe_regs[1][0] = fir_out_reg

        # STAGE 0: Data Acquisition / Alignment
        raw_window = input_queue.get_aligned()
        for k in range(N):
            pipe_regs[0][k] = raw_window[k]            

        current_step += 1
        input_queue.push(raw_signal[signal_idx])
        energy_queue.push(pipe_regs[2][0])
        history_queue.push(pipe_regs[4][0])
        
        signal_idx += 1

        if (trigger_tripped == 1):
            break           

    # FIX: return was inside the while loop (indented too deep)
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


# reference CPU implementation for benchmarking
# cycle model from mini project notebook: 2 cycles per MAC, 1 for squaring etc.
def run_reference_cpu(np.ndarray[DTYPE_t, ndim=1] s_measured, int NL, int Ns, int Ne, int N, int T,
                np.ndarray[DTYPE_t, ndim=1] w1,
                np.ndarray[DTYPE_t, ndim=1] w2, int eq_start, int Max_steps):

    cdef long total_ticks = 0
    cdef long injection_tick = 0
    cdef int t, m
    
    cdef double filtered_val
    cdef double power_val
    cdef double smoothed_val
    
    cdef double sta_accumulator = 0.0
    cdef double lta_accumulator = 0.0
    cdef int sta_ptr = 0
    cdef int lta_ptr = 0

    cdef double[:] bpf_coeffs = w1
    cdef double[:] smooth_coeffs = w2
    cdef double[:] input_signal = s_measured

    cdef np.ndarray energy_history = np.zeros(Ne, dtype=DTYPE)
    cdef np.ndarray sta_window = np.zeros(Ns, dtype=DTYPE)
    cdef np.ndarray lta_window = np.zeros(NL, dtype=DTYPE)

    cdef double[:] en_view = energy_history
    cdef double[:] sta_view = sta_window
    cdef double[:] lta_view = lta_window

    cdef int trigger_flag = 0

    for t in range(Max_steps):

        # BPF - 2 cycles per MAC
        filtered_val = 0.0
        for m in range(N):
            if t - m >= 0:
                filtered_val += bpf_coeffs[m] * input_signal[t - m]
            total_ticks += 2 

        # squaring - 1 cycle
        power_val = filtered_val * filtered_val
        en_view[t % Ne] = power_val
        total_ticks += 1

        # smoothing FIR - 2 cycles per tap
        smoothed_val = 0.0
        for m in range(Ne):
            smoothed_val += smooth_coeffs[m] * en_view[(t - m) % Ne]
            total_ticks += 2

        # O(1) STA/LTA update
        sta_accumulator += (smoothed_val - sta_view[sta_ptr])
        sta_view[sta_ptr] = smoothed_val
        sta_ptr = (sta_ptr + 1) % Ns

        lta_accumulator += (smoothed_val - lta_view[lta_ptr])
        lta_view[lta_ptr] = smoothed_val
        lta_ptr = (lta_ptr + 1) % NL

        total_ticks += 8  # load/store + comparison overhead

        if t == eq_start:
            injection_tick = total_ticks

        if (sta_accumulator * NL > lta_accumulator * T * Ns) and (t > NL):
            trigger_flag = 1
            break

    return {
        'cpu_cycles': total_ticks,
        'eq_start_cycles': injection_tick,
        'detected': trigger_flag == 1,
        'final_step': t
    }
