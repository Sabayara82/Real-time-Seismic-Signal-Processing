# Real-Time-Seismic-Signal-Processing-for-Earthquake-Monitoring-
Real-Time Seismic Signal Processing for Earthquake Monitoring 

## Project Structure

### matlab
```
matlab/
── fir_filter_cascade.m
```
Multi-stage FIR filter design and latency measurement.

### python
```
python/
── pipeline_sim.py
── hazard_analysis.py
```
Pipeline simulator with circular buffers and hazard analysis.

### cython
```
cython/
── mac_loop.pyx
── setup.py
```
Optimized multiply-accumulate kernel and build script.

### analysis
```
analysis/
── precision_test.py
── amdahls_law.py
```
Precision comparison and Amdahl’s Law speedup analysis.

### data
```
data/
── synthetic_seismic.py
```
Generates synthetic seismic signals (P-wave, S-wave, noise).

### report
```
report/
── figures/
```
Figures exported from simulations.