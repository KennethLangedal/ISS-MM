# Python code

With openblas backend, use:
```
export OPENBLAS_NUM_THREADS=1
```
to ensure it is running sequentially.

## Results

| Configuration | N | Time | GFLOPS |
| --- | --- | --- | --- |
| python3 - for | 512 | 58.66616 | 0.00457 |
| pypy3 - for | 512 | 2.04637 | 0.13117 |
| pypy3 - numpy | 512 | 0.01099 | 24.42543 |