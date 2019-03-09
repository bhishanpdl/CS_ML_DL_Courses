# compiler
FC = f95

# library
LIB = -llapack

#FCFLAGS = -g -fbounds-check
#FCFLAGS = -O2
# flags forall (e.g. look for system .mod files, required in gfortran)
#FCFLAGS += -I/usr/include

# libraries needed for linking,
#LDFLAGS = -li_need_this_lib
FFOPTS = -c -O
 
LIBDIR = /usr/local/lib


# List of executables to be built within the package
# If the program depends on other subroutines, the subroutine.o are written separated by space
PROGRAMS = dminv_test

dminv_test: dminv.o print_matrix.o


# run the program
# make run will run the program
run: all
	./dminv_test
	rm -f *.o *~ dminv_test

# "make" builds all
all: $(PROGRAMS)


%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) -L$(LIBDIR)  $(LIB)
	

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -L$(LIBDIR)  $(LIB)

# Utility targets
.PHONY: clean

clean:
	rm -f *.o *.mod *.MOD
	rm -f *~ $(PROGRAMS)
	
	
