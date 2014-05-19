CC = gcc
MPICC = mpicc
MPIFLAGS =
CFLAGS = -I/usr/local/include/gsl #-Wall
LINK = -lgsl -lgslcblas -lm

EXE = GaussSeidel mpiGaussSeidel

all: $(EXE)

%: %.c %.h
	$(CC) $(CFLAGS) -o $@ $< $(LINK)
mpiGaussSeidel: GaussSeidel.o mpiGaussSeidel.o GaussSeidel.h
	$(MPICC) -o $@ $^ $(LINK)
GaussSeidel.o : GaussSeidel.c GaussSeidel.h
	$(MPICC) $(CFLAGS) -c $<
mpiGaussSeidel.o : mpiGaussSeidel.c GaussSeidel.h
	$(MPICC) $(CFLAGS) -c $<

clean:
	rm -f *~ *.o a.out

cleanall:
	rm -f *~ *.o a.out
	rm -f $(EXE)
