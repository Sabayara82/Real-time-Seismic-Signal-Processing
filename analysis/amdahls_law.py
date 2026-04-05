# amdahls_law.py
# amdahl's law analysis for the seismic accelerator project
# shows theoretical speedup limits and where our design sits

import numpy as np
import matplotlib.pyplot as plt
import os

figDir = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')

# our pipeline numbers (from simulation)
NL = 256                # init stall period
totalAccelSteps = 396   # approx total accelerator cycles to detect
cpuCycles = 49530       # cpu cycles from sim
measuredSpeedup = cpuCycles / totalAccelSteps

# serial fraction = stalled init cycles / total cycles
# this is the part that cant be parallelized no matter what
serialFrac = NL / totalAccelSteps
parallelFrac = 1 - serialFrac

print(f"serial fraction (f): {serialFrac:.4f}")
print(f"parallel fraction (1-f): {parallelFrac:.4f}")
print(f"measured speedup: {measuredSpeedup:.2f}x")
print()


# amdahl's law formula
# S(p) = 1 / (f + (1-f)/p)
def amdahl(f, p):
    return 1.0 / (f + (1.0 - f) / p)


# PLOT 1: amdahl's law curves for different serial fractions

numProc = np.arange(1, 129)

serialFracs = [0.01, 0.05, 0.1, 0.25, serialFrac, 0.5, 0.75]

plt.figure(figsize=(11, 6))

for frac in serialFracs:
    speedupVals = [amdahl(frac, p) for p in numProc]
    if abs(frac - serialFrac) < 0.001:
        plt.plot(numProc, speedupVals, linewidth=3, color='red',
                 label=f'f = {frac:.2f} (our design)')
    else:
        plt.plot(numProc, speedupVals, alpha=0.6, label=f'f = {frac:.2f}')

# mark our 8-stage pipeline on the curve
ourSpeedup = amdahl(serialFrac, 8)
plt.scatter([8], [ourSpeedup], color='red', s=150, zorder=5, edgecolors='black',
            label=f'8-stage pipeline (S={ourSpeedup:.1f})')

# theoretical max
maxS = 1.0 / serialFrac
plt.axhline(y=maxS, color='red', linestyle=':', alpha=0.4, label=f'Theoretical max ({maxS:.1f}x)')

plt.xlabel("Number of Parallel Units (p)", fontsize=12)
plt.ylabel("Speedup S(p)", fontsize=12)
plt.title("Amdahl's Law: Speedup vs Parallel Units", fontsize=14)
plt.legend(fontsize=9, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figDir, "amdahls_law_curves.png"), dpi=150)
plt.show()

print(f"theoretical max speedup (p -> inf): {maxS:.2f}x")
print(f"8-stage pipeline theoretical speedup: {ourSpeedup:.2f}x")
print(f"actual measured speedup: {measuredSpeedup:.2f}x")
print(f"(measured is higher because accel has architectural advantages beyond just parallelism)")
print()


# PLOT 2: SIMD width effect on MAC cycles

N_taps = 16    # BPF filter taps
Ne_taps = 10   # smoothing filter taps
fixedOps = 3   # squaring + sta_lta update + detection compare

simdWidths = [1, 2, 4, 8, 16, 32]

cycPerSample = []
for w in simdWidths:
    bpfMACs = int(np.ceil(N_taps / w))
    smoothMACs = int(np.ceil(Ne_taps / w))
    total = bpfMACs + smoothMACs + fixedOps
    cycPerSample.append(total)

baseline_cyc = cycPerSample[0]
simdSpeedups = [baseline_cyc / c for c in cycPerSample]

fig, ax1 = plt.subplots(figsize=(10, 5))

x_positions = np.arange(len(simdWidths))
width = 0.35

color_bar  ='#42A5F5'
color_line = '#E53935'

bars = ax1.bar(x_positions - width/2, cycPerSample, width, color=color_bar, alpha=0.7,
               label='Cycles/sample')
ax1.set_xlabel("SIMD Width", fontsize=12)
ax1.set_ylabel("Cycles per Sample", color=color_bar, fontsize=12)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(simdWidths)

# lets put numbers on bars :)
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.2, f'{int(h)}',
             ha='center', va='bottom', fontsize=10)

ax2 = ax1.twinx()
ax2.plot(x_positions, simdSpeedups, color=color_line, marker='o', linewidth=2.5,
         label='Speedup vs SIMD=1')
ax2.set_ylabel("Speedup", color=color_line, fontsize=12)

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title("Effect of SIMD Width on Pipeline Stage Cycles", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figDir, "simd_width_effect.png"), dpi=150)
plt.show()

print("SIMD width analysis:")
for i, w in enumerate(simdWidths):
    print(f"  SIMD={w:2d}: {cycPerSample[i]:2d} cycles/sample, speedup={simdSpeedups[i]:.2f}x")
print()


# PLOT 3: speedup breakdown - what contributes?

# comparing where the speedup actually comes from
# 1. pipeline parallelism (8 stages overlap)
# 2. SIMD MAC (N=16 taps in fewer cycles)
# 3. register locality (no memory stalls)
# 4. incremental STA/LTA (O(1) vs O(N))

# for the CPU: each sample costs roughly 2*N + 2*Ne + 1 + 4 = 57 base cycles + memory stalls
cpu_base_per_sample = 2 * N_taps + 2 * Ne_taps + 1 + 4
avg_mem_stall = 0.7*1 + 0.2*5 + 0.1*10  # 2.7 cycles average
cpu_with_mem = cpu_base_per_sample + (N_taps + Ne_taps + 1) * avg_mem_stall  # mem access on each MAC + sq

# accelerator: 1 cycle per sample (after init)
accel_per_sample = 1

speedup_from_pipeline = 8  # 8 stages -> 8x from overlap alone (ideally)
speedup_from_simd = N_taps / int(np.ceil(N_taps / N_taps))  # with full SIMD, 1 cycle for all taps
speedup_from_mem = cpu_with_mem / cpu_base_per_sample  # removing memory stalls
speedup_from_incremental = 1  # we use O(1) STA/LTA in both, so no extra speedup here

contribs = {
    'Pipeline\nParallelism': speedup_from_pipeline,
    'SIMD\nMAC': N_taps,  # 16 MACs in 1 cycle vs 16 cycles
    'Register\nLocality': speedup_from_mem,
    'Overall\nMeasured': measuredSpeedup
}

fig, ax = plt.subplots(figsize=(10, 5))
barsX = range(len(contribs))
barVals = list(contribs.values())
barLabels = list(contribs.keys())
colors = ['#66BB6A', '#42A5F5', '#FFA726', '#EF5350']

bars = ax.bar(barsX, barVals, color=colors, alpha=0.8)
ax.set_xticks(barsX)
ax.set_xticklabels(barLabels, fontsize=10)
ax.set_ylabel("Speedup Factor", fontsize=12)
ax.set_title("Speedup Contributions (Approximate)", fontsize=14)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}x',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(figDir, "speedup_breakdown.png"), dpi=150)
plt.show()

print(f"cpu base cycles per sample (no stalls): {cpu_base_per_sample}")
print(f"avg memory stall per access: {avg_mem_stall:.1f} cycles")
print(f"cpu with memory stalls per sample: {cpu_with_mem:.1f}")
print(f"accelerator per sample: {accel_per_sample}")
