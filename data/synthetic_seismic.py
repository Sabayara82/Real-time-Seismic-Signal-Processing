# synthetic_seismic.py
# makes fake earthquake signals for testing the accelerator

import numpy as np
import matplotlib.pyplot as plt
import os

# where the real seismic csvs live
REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'seismic_output')


def makeSignal(signalSize, eqStart=None, seed=42):
    np.random.seed(seed)

    t = np.linspace(0, signalSize, num=signalSize)
    s_norm = np.random.randn(signalSize)

    nComp = 5

    # random noise - supposed to simulate background vibrations
    amps = np.random.uniform(0.01, 20, nComp)
    freqs = np.random.uniform(0.01, 200, nComp)
    phases = np.random.uniform(0, 2*np.pi, nComp)

    for i in range(nComp):
        s_norm += amps[i] * np.sin(2*np.pi*freqs[i]*t + phases[i])

    s_norm = s_norm / np.max(s_norm)

    # earthquake part: lower freq, higher amp
    amps2 = np.random.uniform(0.1, 5, nComp)
    freqs2 = np.random.uniform(0.1, 30, nComp)
    phases2 = np.random.uniform(0, 2*np.pi, nComp)

    eq = np.zeros(signalSize)
    for i in range(nComp):
        eq += amps2[i] * np.sin(2*np.pi*freqs2[i]*t + phases2[i])

    if eqStart is None:
        eqStart = np.random.randint(300, 800)

    eq_sample = np.copy(eq)
    eq_sample[0:eqStart] = 0

    s_measured = s_norm + eq_sample

    return s_measured, eqStart, s_norm, eq


def padSignal(signal, maxSteps):
    # pad with zeros to fill up to maxSteps
    padded = np.append(signal, np.zeros(maxSteps - len(signal)))
    return padded


def loadRealData(channel='BHZ', startSec=-500, endSec=2000, maxSamples=10000):
    # tries to load real Tohoku 2011 data from csv
    # channel can be BHZ, BH1, BH2
    # startSec/endSec are relative to earthquake origin (t=0 is the quake)
    # returns: signal array, sample index where earthquake starts (t=0), sampling rate

    csvPath = os.path.join(REAL_DATA_DIR, f'Tonhoku_2011_{channel}.csv')
    if not os.path.exists(csvPath):
        print(f"real data not found at {csvPath}")
        return None, None, None

    import pandas as pd
    print(f"loading real seismic data from {csvPath}...")
    df = pd.read_csv(csvPath)

    # filter to time window
    mask = (df['seconds_from_origin'] >= startSec) & (df['seconds_from_origin'] <= endSec)
    df_cut = df[mask].reset_index(drop=True)

    # figure out sampling rate from the data
    dt = df_cut['seconds_from_origin'].iloc[1] - df_cut['seconds_from_origin'].iloc[0]
    samplingRate = 1.0 / dt

    signal = df_cut['velocity_m_s'].values
    timeAxis = df_cut['seconds_from_origin'].values

    # find where t=0 is (earthquake origin)
    eqIdx = np.argmin(np.abs(timeAxis))

    # downsample if too many points
    if len(signal) > maxSamples:
        downsampleFactor = len(signal) // maxSamples
        signal = signal[::downsampleFactor]
        timeAxis = timeAxis[::downsampleFactor]
        eqIdx = eqIdx // downsampleFactor
        samplingRate = samplingRate / downsampleFactor
        print(f"  downsampled by {downsampleFactor}x -> {len(signal)} samples, fs={samplingRate:.1f} Hz")

    # normalize so the values are in a reasonable range
    signal = signal / np.max(np.abs(signal))

    print(f"  loaded {len(signal)} samples, eq origin at index {eqIdx}, fs={samplingRate:.1f} Hz")
    return signal, eqIdx, samplingRate


def hasRealData():
    # quick check if the csvs exist
    csvPath = os.path.join(REAL_DATA_DIR, 'Tonhoku_2011_BHZ.csv')
    return os.path.exists(csvPath)


if __name__ == "__main__":
    signal_size = 1000
    sig, eqStart, noise, earthquake = makeSignal(signal_size)

    figDir = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(noise, color='steelblue')
    plt.title("Background Noise (no earthquake)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(earthquake, color='orange')
    plt.title("Raw Earthquake Signal")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(sig, color='green')
    plt.axvline(x=eqStart, color='r', linestyle='--', linewidth=2, label=f'EQ onset @ sample {eqStart}')
    plt.title("Combined Measured Signal (noise + earthquake)")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figDir, "synthetic_signals.png"), dpi=150)
    plt.show()

    print(f"earthquake starts at sample {eqStart}")
    print(f"signal length: {len(sig)}")

    # plot real data if available
    if hasRealData():
        realSig, realEqIdx, realFs = loadRealData(channel='BHZ', startSec=-200, endSec=800)

        if realSig is not None:
            plt.figure(figsize=(12, 4))
            plt.plot(realSig, color='darkred')
            plt.axvline(x=realEqIdx, color='r', linestyle='--', linewidth=2,
                        label=f'EQ origin @ sample {realEqIdx}')
            plt.title("Real Seismic Data: Tohoku 2011 (BHZ channel)")
            plt.ylabel("Normalized Velocity")
            plt.xlabel("Sample Index")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(figDir, "real_seismic_signal.png"), dpi=150)
            plt.show()
            print(f"real data: fs={realFs:.1f} Hz, eq at index {realEqIdx}")
    else:
        print("real seismic data not found - skipping")
