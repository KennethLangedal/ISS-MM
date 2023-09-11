import random, time

random.seed(0)

N = 512

A = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]
B = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]

C = [[0 for _ in range(N)] for _ in range(N)]

t0 = time.time()

for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]

t1 = time.time()
print(t1 - t0)  

cs = 0
for i in range(N):
    for j in range(N):
        cs += C[i][j]

print(cs)
