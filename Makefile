CXX=icpc
CXXFLAGS=-axMIC-AVX512 -qopenmp -mkl -std=c++11 -m64
OPTRPT=-qopt-report=5

default : app


worker.o : worker.cc
	${CXX} -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

app : main.cc worker.o
	${CXX} ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" worker.o

queue: app

	# TIMEIT -m 0x1 app.exe "-0.9" 15 18 4 3 2 4 12 4 4 6 4

clean :
	rm app *.o *.optrpt