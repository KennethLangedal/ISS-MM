import random, time
import numpy as np

random.seed(0)

N = 512

A = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]
B = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]

A = np.array(A)
B = np.array(B)

t0 = time.time()

C = np.matmul(A, B)

t1 = time.time()
print(t1 - t0)  

cs = 0
for i in range(N):
    for j in range(N):
        cs += C[i][j]

print(cs)
