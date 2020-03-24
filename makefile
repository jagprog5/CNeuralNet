CC=cc

all: cneuralnet

clean:
	rm -vf *.o cneuralnet

run: all
	./cneuralnet

# default is visual
run_visual: all
	./cneuralnet 1

run_reduced: all
	./cneuralnet 0

cneuralnet: main.o FFNN.o MNISTRead.o
	$(CC) -o cneuralnet $^ -lm

main.o: main.c FFNN.h MNISTRead.h printReducer.h
	$(CC) -o $@ -c $<

FFNN.o: FFNN.c FFNN.h printReducer.h
	$(CC) -o $@ -c $< -lm

MNISTRead.o: MNISTRead.c MNISTRead.h printReducer.h
	$(CC) -o $@ -c $<