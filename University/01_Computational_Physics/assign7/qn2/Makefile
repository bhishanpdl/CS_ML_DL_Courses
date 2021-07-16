# compiler
FC = f95

# library
LIB = -L/usr/local/lib -llapack

# options
FFOPTS = -c -O
 
# fortran programs
# (note: search and replace file names)
PROGRAMS = hw7qn2 hw7qn2plot

# run the program 1
run1: all
	./hw7qn2
	rm -f *.o *~

# run the program 2
run2: all
	./hw7qn2plot
	rm -f *.o *~

# addidtional program dependencies


# "make" builds all
all: $(PROGRAMS)


%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIB)
	

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< $(LIB)

# Utility targets
.PHONY: clean

clean:
	rm -f *.o *.mod *.MOD
	rm -f *~ $(PROGRAMS)
	
	
