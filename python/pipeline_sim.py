# pipeline_sim.py
# standalone pipeline simulation for the seismic accelerator
# generates all the result plots and metrics

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# adding parent dir so we can import from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from synthetic_seismic import makeSignal, padSignal, loadRealData, hasRealData

figDir = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')


# circular buffer classes
class double_buffer:
    def __init__(self, size):
        self.size = 2 * size
        self.N = size
        self.head = 0
        self.buffer = np.zeros(self.size, dtype=np.float32)

    def push(self, x):
        self.buffer[self.head] = x
        self.buffer[self.head + self.N] = x
        self.head = (self.head + 1) % self.N

    def get_alligned(self):
        return self.buffer[self.head:self.head + self.N]


class energy_buffer:
    def __init__(self, size):
        self.size = size
        self.head = 0
        self.buffer = np.zeros(self.size, dtype=np.float32)

    def push(self, x):
        self.buffer[self.head] = x
        self.head = (self.head + 1) % self.size

    def get_delayed(self, k):
        return self.buffer[(self.head - k) % self.size]


# filter generation

def gen_BPF(fUpper, fLower, fs, N):
    H = np.zeros(N, dtype=complex)
    f_content = np.fft.fftfreq(N, 1/fs)
    for i, f in enumerate(f_content):
        if fLower <= abs(f) <= fUpper:
            H[i] = 1
        else:
            H[i] = 0
    h = np.fft.ifft(H).real
    h = np.roll(h, N//2)
    h = h.astype(np.float32)
    return h


# accelerator sim

def runAccelerator(s_measured, NL, Ns, Ne, N, T, numStages, w1, w2, eqStart, maxSteps):
    in_buff = double_buffer(N)
    energy_buff = double_buffer(Ne)
    filtered_en_buff = energy_buffer(NL)
    acc_pipeline = np.zeros((numStages, NL), dtype=np.float32)

    STA = 0.0
    LTA = 0.0
    stalled = 0
    detection = 0
    step = 0
    i = 0

    LTA_vals = []
    STA_vals = []
    ltaAvg = []
    staAvg = []

    while step < maxSteps:
        if step < NL:
            stalled += 1

        # stage 8 - comparison
        if (STA * NL > LTA * T * Ns) and (i > NL):
            detection = 1
            acc_pipeline[7] = np.ones_like(acc_pipeline[7])
            acc_pipeline[7][0] = LTA
            acc_pipeline[7][1] = STA

        # stage 7 - STA/LTA update
        LTA += filtered_en_buff.buffer[(filtered_en_buff.head - 1) % filtered_en_buff.size] - filtered_en_buff.get_delayed(NL)
        STA += filtered_en_buff.buffer[(filtered_en_buff.head - 1) % filtered_en_buff.size] - filtered_en_buff.get_delayed(Ns)
        LTA_vals.append(LTA)
        STA_vals.append(STA)
        ltaAvg.append(LTA / NL)
        staAvg.append(STA / Ns)

        acc_pipeline[6][0] = STA
        acc_pipeline[6][1] = LTA

        # stage 6
        acc_pipeline[5] = filtered_en_buff.buffer

        # stage 5 - smoothing
        acc_pipeline[4][0] = np.sum(acc_pipeline[3][:Ne] * w2)

        # stage 4 - energy buffer
        acc_pipeline[3][:Ne] = energy_buff.get_alligned()

        # stage 3 - squaring
        acc_pipeline[2][0] = np.square(acc_pipeline[1][0])

        # stage 2 - bandpass FIR
        acc_pipeline[1][0] = np.sum(acc_pipeline[0][:N] * w1)

        # stage 1 - input buffer
        acc_pipeline[0][:N] = in_buff.get_alligned()
        step += 1

        in_buff.push(s_measured[i])
        energy_buff.push(acc_pipeline[2][0])
        filtered_en_buff.push(acc_pipeline[4][0])
        i += 1

        if detection == 1:
            break

    return {
        'totalSteps': step,
        'stalled': stalled,
        'detected': detection == 1,
        'LTA_vals': LTA_vals,
        'STA_vals': STA_vals,
        'ltaAvg': ltaAvg,
        'staAvg': staAvg,
    }


# CPU baseline sim

def mem_access_cost():
    r = np.random.random()
    if r < 0.7:
        return 1
    if r < 0.9:
        return 5
    return 10

def runCPU(s_measured, NL, Ns, Ne, N, T, w1, w2, eqStart, maxSteps):
    cpuCycles = 0
    instructions = 0
    e_buffer = np.zeros(Ne)
    STA1 = 0.0
    LTA1 = 0.0
    sta_buffer = np.zeros(Ns)
    lta_buffer = np.zeros(NL)
    sta_idx = 0
    lta_idx = 0
    eq_start_cycles = 0
    detected = False

    # tracking where cycles go
    bpf_cycles = 0
    smooth_cycles = 0
    stalta_cycles = 0

    for j in range(maxSteps):
        # BPF filter
        y = 0
        for k in range(N):
            if j - k >= 0:
                y += w1[k] * s_measured[j - k]
                cost = 2 + mem_access_cost()
                cpuCycles += cost
                bpf_cycles += cost
                instructions += 2

        # squaring
        e = y * y
        e_buffer[j % Ne] = e
        cpuCycles += 1 + mem_access_cost()
        instructions += 1

        # smoothing filter
        s = 0
        for k in range(Ne):
            idx = (j - k) % Ne
            s += w2[k] * e_buffer[idx]
            cost = 2 + mem_access_cost()
            cpuCycles += cost
            smooth_cycles += cost
            instructions += 2

        old_sta = sta_buffer[sta_idx]
        old_lta = lta_buffer[lta_idx]
        STA1 += s - old_sta
        LTA1 += s - old_lta
        sta_buffer[sta_idx] = s
        lta_buffer[lta_idx] = s
        sta_idx = (sta_idx + 1) % Ns
        lta_idx = (lta_idx + 1) % NL

        cpuCycles += 4
        stalta_cycles += 4
        instructions += 4

        if j == eqStart:
            eq_start_cycles = cpuCycles

        if (STA1 * NL > LTA1 * T * Ns) and (j > NL):
            detected = True
            break

    return {
        'cpuCycles': cpuCycles,
        'instructions': instructions,
        'eq_start_cycles': eq_start_cycles,
        'detected': detected,
        'finalStep': j,
        'bpf_cycles': bpf_cycles,
        'smooth_cycles': smooth_cycles,
        'stalta_cycles': stalta_cycles,
    }



if __name__ == "__main__":
    np.random.seed(42)

    # params
    Max_steps = 10000
    signal_size = 1000
    NL = 256
    Ns = 8
    Ne = 10
    N = 16
    T = 2**2
    f_upper = 30
    f_lower = 0.1
    fs = 200
    numStages = 8

    # generate signal
    sig_raw, eq_start, _, _ = makeSignal(signal_size, seed=42)
    s_measured = padSignal(sig_raw, Max_steps)

    # filter weights
    w1 = gen_BPF(f_upper, f_lower, fs, N)
    w1 = w1[::-1]
    w2 = np.array([0.8**k for k in range(1, Ne+1)])
    w2 = w2 / np.sum(w2)
    w2 = w2.astype(np.float32)

    print("="*50)
    print("  SEISMIC ACCELERATOR SIMULATION")
    print("="*50)
    print(f"Signal size: {signal_size}, Max steps: {Max_steps}")
    print(f"NL={NL}, Ns={Ns}, Ne={Ne}, N={N}, T={T}")
    print(f"Earthquake injected at sample {eq_start}")
    print()

    # run accelerator
    print("Running accelerator simulation...")
    t_start = time.perf_counter()
    accelResult = runAccelerator(s_measured, NL, Ns, Ne, N, T, numStages, w1, w2, eq_start, Max_steps)
    accel_time = time.perf_counter() - t_start
    print(f"  done in {accel_time:.4f}s")

    # run CPU baseline
    print("Running CPU baseline...")
    t_start = time.perf_counter()
    cpuResult = runCPU(s_measured, NL, Ns, Ne, N, T, w1, w2, eq_start, Max_steps)
    cpu_time = time.perf_counter() - t_start
    print(f"  done in {cpu_time:.4f}s")

    # print metrics
    accelCycles = accelResult['totalSteps']
    cpuCycles = cpuResult['cpuCycles']
    speedup = cpuCycles / accelCycles
    accel_detection_latency = accelCycles - eq_start
    cpu_detection_latency = cpuCycles - cpuResult['eq_start_cycles']

    useful = accelCycles - accelResult['stalled']
    accelThroughput = useful / accelCycles
    accelCPI = 1.0 / accelThroughput if accelThroughput > 0 else float('inf')
    cpuCPI = cpuCycles / cpuResult['instructions']
    cpuThroughput = cpuResult['instructions'] / cpuCycles

    print()
    print("-"*50)
    print("ACCELERATOR METRICS")
    print(f"  Total cycles: {accelCycles}")
    print(f"  Stalled cycles (init): {accelResult['stalled']}")
    print(f"  Useful cycles: {useful}")
    print(f"  Throughput (overall): {accelThroughput:.4f}")
    print(f"  Throughput (post-init): 1.0")
    print(f"  CPI (overall): {accelCPI:.4f}")
    print(f"  CPI (post-init): 1.0")
    print(f"  Efficiency: {(useful/accelCycles)*100:.1f}%")
    print(f"  Detection latency: {accel_detection_latency} cycles after EQ")
    print()
    print("CPU BASELINE METRICS")
    print(f"  Total cycles: {cpuCycles}")
    print(f"  Instructions: {cpuResult['instructions']}")
    print(f"  CPI: {cpuCPI:.4f}")
    print(f"  Throughput (IPC): {cpuThroughput:.4f}")
    print(f"  Detection latency: {cpu_detection_latency} cycles after EQ")
    print()
    print(f"SPEEDUP: {speedup:.2f}x")
    print(f"Python wall-clock: accel={accel_time:.4f}s, cpu={cpu_time:.4f}s")
    print("-"*50)

    # PLOT 1: STA vs LTA with detection markers
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].set_title("LTA vs STA (raw sums)")
    axes[0].plot(accelResult['LTA_vals'], label='LTA', alpha=0.8)
    axes[0].plot(accelResult['STA_vals'], label='STA', alpha=0.8)
    axes[0].axvline(x=eq_start, color='r', linestyle='--', linewidth=2, label=f'EQ start @ {eq_start}')
    axes[0].axvline(x=NL, color='y', linestyle='--', linewidth=2, label='Init period end')
    axes[0].axvline(x=accelCycles, color='g', linestyle='--', linewidth=2, label=f'Detection @ {accelCycles}')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Energy Sum")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Average LTA vs Average STA")
    axes[1].plot(accelResult['ltaAvg'], label='LTA/NL', alpha=0.8)
    axes[1].plot(accelResult['staAvg'], label='STA/Ns', alpha=0.8)
    axes[1].axvline(x=eq_start, color='r', linestyle='--', linewidth=2, label=f'EQ start @ {eq_start}')
    axes[1].axvline(x=NL, color='y', linestyle='--', linewidth=2, label='Init period end')
    axes[1].axvline(x=accelCycles, color='g', linestyle='--', linewidth=2, label=f'Detection @ {accelCycles}')
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Average Energy")
    axes[1].set_xlabel("Cycle")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "sta_lta_detection.png"), dpi=150)
    plt.show()

    # PLOT 2: cycle breakdown comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # bar chart - total cycles
    bars = axes[0].bar(['Accelerator', 'CPU'], [accelCycles, cpuCycles],
                       color=['#2196F3', '#FF5722'])
    axes[0].set_ylabel("Total Cycles")
    axes[0].set_title("Total Cycles to Detection")
    # put numbers on bars
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                     ha='center', va='bottom', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # cpu cycle breakdown pie chart
    other_cycles = cpuCycles - cpuResult['bpf_cycles'] - cpuResult['smooth_cycles'] - cpuResult['stalta_cycles']
    labels = ['BPF MAC', 'Smoothing MAC', 'STA/LTA', 'Other (sq + mem)']
    sizes = [cpuResult['bpf_cycles'], cpuResult['smooth_cycles'], cpuResult['stalta_cycles'], other_cycles]
    colors_pie = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("CPU Cycle Breakdown")

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "cycle_breakdown.png"), dpi=150)
    plt.show()

    # PLOT 3: CPI and throughput comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CPI
    cpiVals = [1.0, accelCPI, cpuCPI]
    cpiLabels = ['Accel (post-init)', 'Accel (overall)', 'CPU']
    barColors = ['#4CAF50', '#8BC34A', '#FF5722']
    bars = axes[0].bar(cpiLabels, cpiVals, color=barColors)
    axes[0].set_ylabel("CPI")
    axes[0].set_title("Cycles Per Instruction")
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}',
                     ha='center', va='bottom', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # throughput
    thrVals = [1.0, accelThroughput, cpuThroughput]
    thrLabels = ['Accel (post-init)', 'Accel (overall)', 'CPU']
    bars2 = axes[1].bar(thrLabels, thrVals, color=barColors)
    axes[1].set_ylabel("Instructions Per Cycle (IPC)")
    axes[1].set_title("Throughput")
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}',
                     ha='center', va='bottom', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "cpi_throughput.png"), dpi=150)
    plt.show()

    # PLOT 4: throughput over time (accelerator)
    # cumulative useful outputs vs cycle
    totalCycles_arr = np.arange(1, accelCycles + 1)
    usefulOutputs = np.zeros(accelCycles)
    for c in range(accelCycles):
        if c >= NL:
            usefulOutputs[c] = c - NL + 1
        else:
            usefulOutputs[c] = 0

    instantThroughput = np.zeros(accelCycles)
    for c in range(1, accelCycles):
        if c >= NL:
            instantThroughput[c] = 1.0  # 1 output per cycle after init
        else:
            instantThroughput[c] = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(totalCycles_arr, usefulOutputs, color='#2196F3')
    axes[0].axvline(x=NL, color='y', linestyle='--', label=f'Init done (cycle {NL})')
    axes[0].set_title("Cumulative Useful Outputs vs Cycle")
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("Useful Outputs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(totalCycles_arr, instantThroughput, color='#4CAF50')
    axes[1].axvline(x=NL, color='y', linestyle='--', label=f'Init done (cycle {NL})')
    axes[1].set_title("Instantaneous Throughput (outputs/cycle)")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Throughput")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "throughput_over_time.png"), dpi=150)
    plt.show()

    # PLOT 5: detection latency histogram (many runs)
    print("\nRunning multiple seeds for detection latency histogram...")
    latencies = []
    numTrials = 50
    for trial in range(numTrials):
        trialSeed = trial + 100
        trialSig, trialEqStart, _, _ = makeSignal(signal_size, seed=trialSeed)
        trialSig = padSignal(trialSig, Max_steps)

        res = runAccelerator(trialSig, NL, Ns, Ne, N, T, numStages, w1, w2, trialEqStart, Max_steps)
        if res['detected']:
            lat = res['totalSteps'] - trialEqStart
            latencies.append(lat)

    plt.figure(figsize=(10, 5))
    plt.hist(latencies, bins=20, color='#2196F3', edgecolor='black', alpha=0.8)
    plt.axvline(x=np.mean(latencies), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(latencies):.1f} cycles')
    plt.xlabel("Detection Latency (cycles after EQ onset)")
    plt.ylabel("Count")
    plt.title(f"Detection Latency Distribution ({numTrials} trials)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "detection_latency_hist.png"), dpi=150)
    plt.show()

    print(f"  avg latency: {np.mean(latencies):.1f} cycles")
    print(f"  std latency: {np.std(latencies):.1f} cycles")
    print(f"  min: {np.min(latencies)}, max: {np.max(latencies)}")

    # PLOT 6: threshold sweep
    print("\nRunning threshold sweep...")
    thresholdVals = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    thresh_latencies = []
    thresh_detected = []

    for tVal in thresholdVals:
        res = runAccelerator(s_measured, NL, Ns, Ne, N, tVal, numStages, w1, w2, eq_start, Max_steps)
        thresh_detected.append(res['detected'])
        if res['detected']:
            thresh_latencies.append(res['totalSteps'] - eq_start)
        else:
            thresh_latencies.append(Max_steps)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#2196F3'
    ax1.bar(range(len(thresholdVals)), thresh_latencies, color=color1, alpha=0.7)
    ax1.set_xlabel("Threshold (T)")
    ax1.set_ylabel("Detection Latency (cycles)", color=color1)
    ax1.set_xticks(range(len(thresholdVals)))
    ax1.set_xticklabels(thresholdVals)
    ax1.set_title("Detection Latency vs Threshold Value")
    ax1.grid(True, alpha=0.3, axis='y')

    # mark which ones didnt detect
    for i, det in enumerate(thresh_detected):
        if not det:
            ax1.text(i, thresh_latencies[i], 'NO\nDET', ha='center', va='bottom',
                     color='red', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "threshold_sweep.png"), dpi=150)
    plt.show()

    # PLOT 7: window size (NL) sweep
    print("\nRunning NL window sweep...")
    nlVals = [32, 64, 128, 256, 512]
    nl_latencies = []
    nl_initCycles = []
    nl_totalCycles = []

    for nlVal in nlVals:
        res = runAccelerator(s_measured, nlVal, Ns, Ne, N, T, numStages, w1, w2, eq_start, Max_steps)
        nl_totalCycles.append(res['totalSteps'])
        nl_initCycles.append(res['stalled'])
        if res['detected']:
            nl_latencies.append(res['totalSteps'] - eq_start)
        else:
            nl_latencies.append(-1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(nlVals, nl_initCycles, marker='o', color='#FF5722', linewidth=2)
    axes[0].set_xlabel("NL (LTA window size)")
    axes[0].set_ylabel("Initialization Cycles")
    axes[0].set_title("Init Time vs LTA Window Size")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(nlVals)), nl_latencies, color='#4CAF50', alpha=0.7)
    axes[1].set_xlabel("NL (LTA window size)")
    axes[1].set_ylabel("Detection Latency (cycles)")
    axes[1].set_xticks(range(len(nlVals)))
    axes[1].set_xticklabels(nlVals)
    axes[1].set_title("Detection Latency vs LTA Window Size")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "window_size_sweep.png"), dpi=150)
    plt.show()

    # PLOT 8: scalability - execution time vs signal length
    print("\nRunning scalability analysis...")
    signalLengths = [200, 500, 1000, 2000, 3000, 5000]
    accel_times_list = []
    cpu_times_list = []

    for sLen in signalLengths:
        testSig, testEq, _, _ = makeSignal(sLen, seed=42)
        testSig = padSignal(testSig, sLen + 2000)
        testMaxSteps = sLen + 2000

        t0 = time.perf_counter()
        runAccelerator(testSig, NL, Ns, Ne, N, T, numStages, w1, w2, testEq, testMaxSteps)
        accel_times_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        runCPU(testSig, NL, Ns, Ne, N, T, w1, w2, testEq, testMaxSteps)
        cpu_times_list.append(time.perf_counter() - t0)

    plt.figure(figsize=(10, 5))
    plt.plot(signalLengths, accel_times_list, marker='o', label='Accelerator sim', linewidth=2)
    plt.plot(signalLengths, cpu_times_list, marker='s', label='CPU sim', linewidth=2)
    plt.xlabel("Signal Length (samples)")
    plt.ylabel("Wall-clock Time (seconds)")
    plt.title("Scalability: Execution Time vs Signal Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "scalability.png"), dpi=150)
    plt.show()