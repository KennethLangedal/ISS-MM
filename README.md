# ISS-MM
ISS lecture at ICT meeting about matrix multiplication

CPU frequency commands.
```
sudo cpupower frequency-set -d 2000mhz
sudo cpupower frequency-set -u 2000mhz
```

Compile Slides.
```
pdflatex -output-directory bin main.tex
```

OpenBLAS number of threads.
```
export OPENBLAS_NUM_THREADS=1
```

Profiling commands.
```
sudo sysctl kernel.perf_event_paranoid=<parameter>
sudo sysctl kernel.kptr_restrict=<parameter>
```