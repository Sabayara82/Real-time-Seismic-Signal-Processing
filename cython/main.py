#THIS FILE WAS GENERATED FOR TESTING ONLY - DO NOT UPLOAD!

import numpy as np
import project  # or whatever your .pyx file is named (without extension)

# set up some test inputs
fs = 100.0        # sampling frequency
NL = 200          # long window
Ns = 20           # short window  
Ne = 10           # energy smoothing window
N = 32            # FIR taps
T = 3             # detection threshold

# generate a fake signal with an earthquake injected halfway through
np.random.seed(42)
eq_start = 500
s = np.random.randn(2000).astype(np.float64) * 0.1
s[eq_start:] += np.random.randn(1500).astype(np.float64) * 2.0  # big spike

# get filter coefficients
w1 = project.gen_BPF(10.0, 1.0, fs, N)
w2 = np.ones(Ne, dtype=np.float64) / Ne  # simple averaging smoother

# run it
result = project.run_accelerator(s, NL, Ns, Ne, N, T,
                                     8, w1, w2, eq_start, len(s))

print("Detected:", result['is_detected'])
print("Triggered at step:", result['total_steps'])
print("Stalled cycles:", result['stalled_cycles'])
print("Earthquake injected at:", result['eq_start_idx'])

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(result['lta_averages'], label='LTA')
plt.plot(result['sta_averages'], label='STA')
plt.axvline(eq_start, color='r', linestyle='--', label='eq injected')
plt.axvline(result['total_steps'], color='g', linestyle='--', label='triggered')
plt.legend()
plt.title('STA vs LTA')
plt.show()
