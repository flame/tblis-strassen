SCALAPACK_LIBS = -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64

INSTALL_PATH = ${HOME}/lib

MPICXX = mpicxx
#MPICC = mpicc

EXEC = tensorblis.exe tensor_1_abc.exe tensor_1_ab.exe tensor_1_naive.exe tensor_2_abc.exe tensor_2_ab.exe tensor_2_naive.exe

SRC = $(wildcard *.cxx)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CFLAGS = -O3 -Wall -openmp -mkl -std=c++11  
CFLAGS_TBLIS   = ${CFLAGS} -DTBLIS
CFLAGS_STR1ABC = ${CFLAGS} -DSTR1ABC
CFLAGS_STR1AB  = ${CFLAGS} -DSTR1AB
CFLAGS_STR1N   = ${CFLAGS} -DSTR1N
CFLAGS_STR2ABC = ${CFLAGS} -DSTR2ABC
CFLAGS_STR2AB  = ${CFLAGS} -DSTR2AB
CFLAGS_STR2N   = ${CFLAGS} -DSTR2N

INC = -I./                  \
	  -I${INSTALL_PATH}/include/tblis \
	  -I${INSTALL_PATH}/include \
	  -I${MKLROOT}/include \
	  -I../src \

LDFLAGS = -L${INSTALL_PATH}/lib -L${MKLROOT}/lib/intel64

LIBS = ${SCALAPACK_LIBS} -ltblis -ltci

all: ${EXEC} 

tensorblis.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_TBLIS}  ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_1_abc.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR1ABC}  ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_1_ab.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR1AB}   ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_1_naive.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR1N}    ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_2_abc.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR2ABC}  ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_2_ab.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR2AB}   ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

tensor_2_naive.exe: $(SRC) Makefile
	$(MPICXX) ${CFLAGS_STR2N}    ${INC} ${LDFLAGS} $(SRC) -o $@  ${LIBS}

clean:
	rm -f *.o *.s *.d *~ ${EXEC}

