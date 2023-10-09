# Matrix multiplication code for ISS talk

Compile using make and the name of the configuration you want to compile.
For example, compiling the for loop version:

```
make MM_for
```

Run using:

```
./a.out M N K it
```

Where M, N, and K are the matrix dimentions, and it the number of iterations.

The kernel and block_kernel requires "nice" dimentions.

Block requires M to be multiple of 4 and N to be multiple of 16.

Block_kernel requires M to be multiple of 128, N to be multiple of 16, and K to be multiple of 256.