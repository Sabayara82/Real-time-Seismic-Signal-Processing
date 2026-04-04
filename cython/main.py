#THIS FILE WAS GENERATED FOR TESTING ONLY - DO NOT UPLOAD!

# ENCM 515 Final Project - Test Suite
# Validates the Cython accelerator output against the reference Python simulation
# Group 8

import numpy as np
import matplotlib.pyplot as plt
import project  # the compiled .pyd from project.pyx


# ------------------------------------------------------------------ #
#  Shared parameters - must match what both simulations use
# ------------------------------------------------------------------ #
np.random.seed(42)

Max_steps   = 10000
signal_size = 1000

NL = 256
Ns = 8
Ne = 10
num_acc_stages = 8

f_upper = 30
f_lower = 0.1
fs      = 200
N       = 16

T = 2**2


# ------------------------------------------------------------------ #
#  Generate the same signal used in the reference sim
# ------------------------------------------------------------------ #
t      = np.linspace(0, signal_size, num=signal_size)
s_norm = np.random.randn(signal_size)

n_comp = 5
amps   = np.random.uniform(0.01, 20,       n_comp)
freqs  = np.random.uniform(0.01, 200,      n_comp)
phases = np.random.uniform(0,    2*np.pi,  n_comp)
for i in range(n_comp):
    s_norm += amps[i] * np.sin(2*np.pi*freqs[i]*t + phases[i])
s_norm = s_norm / np.max(s_norm)

amps   = np.random.uniform(0.1, 5,        n_comp)
freqs  = np.random.uniform(0.1, 30,       n_comp)
phases = np.random.uniform(0,   2*np.pi,  n_comp)
eq = np.zeros(signal_size)
for i in range(n_comp):
    eq += amps[i] * np.sin(2*np.pi*freqs[i]*t + phases[i])

eq_start = np.random.randint(300, 800)
eq_sample = np.copy(eq)
eq_sample[0:eq_start] = 0
s_measured = s_norm + eq_sample

s_measured = np.append(s_measured, np.zeros(Max_steps - signal_size))
s_measured = s_measured.astype(np.float64)   # Cython version uses float64


# ------------------------------------------------------------------ #
#  Build filter weights the same way the reference sim does
# ------------------------------------------------------------------ #
def gen_BPF_ref(f_upper, f_lower, fs, N):
    H = np.zeros(N, dtype=complex)
    f_content = np.fft.fftfreq(N, 1/fs)
    for i, f in enumerate(f_content):
        H[i] = 1 if f_lower <= abs(f) <= f_upper else 0
    h = np.fft.ifft(H).real
    h = np.roll(h, N//2)
    return h.astype(np.float64)

w1_ref = gen_BPF_ref(f_upper, f_lower, fs, N)
w1_ref = w1_ref[::-1]   # reference sim flips the coefficients

w2 = np.array([0.8**k for k in range(1, Ne+1)], dtype=np.float64)
w2 = w2 / np.sum(w2)


# ------------------------------------------------------------------ #
#  Reference Python simulation (copied directly from your notebook)
# ------------------------------------------------------------------ #
class double_buffer:
    def __init__(self, size):
        self.size = 2 * size
        self.N    = size
        self.head = 0
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def push(self, x):
        self.buffer[self.head]          = x
        self.buffer[self.head + self.N] = x
        self.head = (self.head + 1) % self.N

    def get_alligned(self):
        return self.buffer[self.head : self.head + self.N]


class energy_buffer:
    def __init__(self, size):
        self.size   = size
        self.head   = 0
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def push(self, x):
        self.buffer[self.head] = x
        self.head = (self.head + 1) % self.size

    def get_delayed(self, k):
        return self.buffer[(self.head - k) % self.size]


def run_reference(s_measured, w1, w2):
    in_buff         = double_buffer(N)
    energy_buff     = double_buffer(Ne)
    filtered_en_buff = energy_buffer(NL)
    acc_pipeline    = np.zeros((num_acc_stages, NL), dtype=np.float64)

    STA, LTA  = 0.0, 0.0
    stalled   = 0
    detection = 0
    step      = 0
    i         = 0

    LTA_vals, STA_vals = [], []
    LTA_ave_vals, STA_ave_vals = [], []

    while step < Max_steps:
        if step < NL:
            stalled += 1

        if (STA * NL > LTA * T * Ns) and (i > NL):
            detection = 1
            acc_pipeline[7]    = np.ones_like(acc_pipeline[7])
            acc_pipeline[7][0] = LTA
            acc_pipeline[7][1] = STA

        LTA += filtered_en_buff.buffer[(filtered_en_buff.head - 1) % filtered_en_buff.size] - filtered_en_buff.get_delayed(NL)
        STA += filtered_en_buff.buffer[(filtered_en_buff.head - 1) % filtered_en_buff.size] - filtered_en_buff.get_delayed(Ns)
        LTA_vals.append(LTA);       STA_vals.append(STA)
        LTA_ave_vals.append(LTA/NL); STA_ave_vals.append(STA/Ns)

        acc_pipeline[6][0] = STA
        acc_pipeline[6][1] = LTA
        acc_pipeline[5]    = filtered_en_buff.buffer
        acc_pipeline[4][0] = np.sum(acc_pipeline[3][:Ne] * w2)
        acc_pipeline[3][:Ne] = energy_buff.get_alligned()
        acc_pipeline[2][0] = np.square(acc_pipeline[1][0])
        acc_pipeline[1][0] = np.sum(acc_pipeline[0][:N] * w1)
        acc_pipeline[0][:N] = in_buff.get_alligned()

        step += 1
        in_buff.push(s_measured[i])
        energy_buff.push(acc_pipeline[2][0])
        filtered_en_buff.push(acc_pipeline[4][0])
        i += 1

        if detection == 1:
            break

    return {
        'total_steps'   : step,
        'stalled_cycles': stalled,
        'is_detected'   : detection == 1,
        'lta_history'   : LTA_vals,
        'sta_history'   : STA_vals,
        'lta_averages'  : LTA_ave_vals,
        'sta_averages'  : STA_ave_vals,
    }


# ------------------------------------------------------------------ #
#  Run both simulations
# ------------------------------------------------------------------ #
print("Running reference Python simulation...")
ref = run_reference(s_measured, w1_ref, w2)

print("Running Cython accelerator...")
w1_cython = project.gen_BPF(float(f_upper), float(f_lower), float(fs), N)
w1_cython = w1_cython[::-1]
cython_result = project.run_accelerator(
    s_measured, NL, Ns, Ne, N, T,
    num_acc_stages, w1_cython, w2, eq_start, Max_steps
)


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


print("\n--- Detection ---")
check("Both simulations detect earthquake",
      ref['is_detected'] and cython_result['is_detected'])

check("Detection step within 20 cycles of reference",
      abs(cython_result['total_steps'] - ref['total_steps']) <= 20,
      f"ref={ref['total_steps']}  cython={cython_result['total_steps']}")

print("\n--- Stall count ---")
check("Stalled cycles == NL",
      cython_result['stalled_cycles'] == NL,
      f"got {cython_result['stalled_cycles']}, expected {NL}")

check("Ref stalled cycles == NL",
      ref['stalled_cycles'] == NL,
      f"got {ref['stalled_cycles']}, expected {NL}")

print("\n--- STA/LTA shape ---")
n = min(len(ref['lta_averages']), len(cython_result['lta_averages']))
lta_ref    = np.array(ref['lta_averages'][:n])
lta_cython = np.array(cython_result['lta_averages'][:n])
sta_ref    = np.array(ref['sta_averages'][:n])
sta_cython = np.array(cython_result['sta_averages'][:n])

check("LTA averages close to reference (tol=0.1)",
      np.allclose(lta_ref, lta_cython, atol=0.1),
      f"max diff = {np.max(np.abs(lta_ref - lta_cython)):.4f}")

check("STA averages close to reference (tol=0.1)",
      np.allclose(sta_ref, sta_cython, atol=0.1),
      f"max diff = {np.max(np.abs(sta_ref - sta_cython)):.4f}")

check("STA spikes after earthquake injection",
      np.mean(sta_cython[eq_start+10:]) > np.mean(sta_cython[NL:eq_start]),
      "STA did not increase after earthquake")

print("\n--- Detection latency ---")
latency = cython_result['total_steps'] - eq_start
check("Detection latency is positive",
      latency > 0,
      f"latency = {latency}")
check("Detection latency is under 100 cycles",
      latency < 100,
      f"latency = {latency} cycles")
print(f"        Detection latency: {latency} cycles after eq_start={eq_start}")

print(f"\n{'='*40}")
print(f"  {passed} passed,  {failed} failed")
print(f"{'='*40}\n")


# ------------------------------------------------------------------ #
#  Plot - STA/LTA comparison between reference and Cython
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

axes[0].set_title("LTA average - reference vs Cython")
axes[0].plot(lta_ref,    label="Reference", alpha=0.7)
axes[0].plot(lta_cython, label="Cython",    alpha=0.7, linestyle='--')
axes[0].axvline(eq_start,                    color='r', linestyle=':', label=f'eq injected @ {eq_start}')
axes[0].axvline(cython_result['total_steps'], color='g', linestyle=':', label=f'Cython trigger @ {cython_result["total_steps"]}')
axes[0].axvline(ref['total_steps'],           color='b', linestyle=':', label=f'Ref trigger @ {ref["total_steps"]}')
axes[0].legend(fontsize=8)

axes[1].set_title("STA average - reference vs Cython")
axes[1].plot(sta_ref,    label="Reference", alpha=0.7)
axes[1].plot(sta_cython, label="Cython",    alpha=0.7, linestyle='--')
axes[1].axvline(eq_start,                    color='r', linestyle=':')
axes[1].axvline(cython_result['total_steps'], color='g', linestyle=':')
axes[1].axvline(ref['total_steps'],           color='b', linestyle=':')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("sta_lta_comparison.png", dpi=150)
plt.show()
print("Plot saved to sta_lta_comparison.png")gered')
plt.legend()
plt.title('STA vs LTA')
plt.show()
