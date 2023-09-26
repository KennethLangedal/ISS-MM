compile = gcc -std=gnu17 -O3 -march=native -Wall -ffast-math -g

%: %.c
	$(compile) -c $<
	$(compile) -c main.c
	gcc main.o $@.o -lopenblas

clean:
	rm -rf *.o *.out