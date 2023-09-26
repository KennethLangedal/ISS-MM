compile = gcc -std=gnu17 -O3 -march=native -Wall -ffast-math -fopenmp

%: %.c
	$(compile) -c $<
	$(compile) -c main.c
	gcc main.o $@.o -lopenblas -fopenmp

clean:
	rm -rf *.o *.out