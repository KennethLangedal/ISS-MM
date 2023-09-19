compile = gcc -std=gnu17 -O3 -march=native -Wall -ffast-math

%: %.c
	$(compile) -c $<
	$(compile) -c main.c
	gcc main.o $@.o

clean:
	rm -rf *.o *.out