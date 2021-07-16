
FC = f95

#FCFLAGS = -g -fbounds-check
#FCFLAGS = -O2
# flags forall (e.g. look for system .mod files, required in gfortran)
#FCFLAGS += -I/usr/include

# libraries needed for linking,
#LDFLAGS = -li_need_this_lib
FFOPTS = -c -O
 
LIBDIR = /usr/local/lib


# List of executables to be built within the package
PROGRAMS = dgesv_test

dgesv_test: dgesv.o         # dgesv.f90 is subroutine

# "make" builds all
all: $(PROGRAMS)



%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) -L$(LIBDIR)  -llapack
	

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -L$(LIBDIR)  -llapack

# Utility targets
.PHONY: clean veryclean

clean:
	rm -f *.o *.mod *.MOD

veryclean: clean
	rm -f *~ $(PROGRAMS)

