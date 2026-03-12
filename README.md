# Real-Time-Seismic-Signal-Processing-for-Earthquake-Monitoring-
Real-Time Seismic Signal Processing for Earthquake Monitoring 


seismic-accelerator/
├── matlab/
│   └── fir_filter_cascade.m       --> multi-stage filter design + latency measurement
├── python/
│   ├── pipeline_sim.py            --> circular buffer + pipeline stage simulator
│   └── hazard_analysis.py        --> data hazards, initiation interval
├── cython/
│   ├── mac_loop.pyx               --> optimized mac kernel
│   └── setup.py                   --> build script
├── analysis/
│   ├── precision_test.py          --> float32 vs fixed-point Q15/Q31 comparison
│   └── amdahls_law.py             --> speedup ceiling plots
├── data/
│   └── synthetic_seismic.py      --> generates test signals (P-wave, S-wave, noise)
│   └── synthetic_seismic.py
└── output/
    └── figures/                   --> output plots exported from simulations