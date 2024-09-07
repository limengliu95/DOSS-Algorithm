IFLAG = -I
INCLUDE=

## works on Mac OSX v 10.5.6 with ifort 10.1.017
#F90C = ifort
#FFLAGS = -O2 -Vaxlib -cm -openmp -w -vec_report0 -ip -xP

##Gfortran compiler Mac OSX
F90C     = gfortran
FFLAGS = -O3 -ffixed-line-length-none -ffree-line-length-none -fopenmp

OBJFILES = getlrgdr7like.o utilsforlrglike.o

default: getlrgdr7like

all: getlrgdr7like

getlrgdr7like.o: utilsforlrglike.o

%.o: %.f90
	$(F90C) $(F90FLAGS) -c $*.f90

%.o: %.F90
	$(F90C) $(F90FLAGS) -c $*.F90

getlrgdr7like: $(OBJFILES)
	$(F90C) -o getlrgdr7like $(OBJFILES) $(F90FLAGS)

clean:
	rm -f *.o *.mod
