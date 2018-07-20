CXX=icl
CXXFLAGS=-QaxMIC-AVX512 -Qopenmp -Qmkl -Qstd=c++11 -Qm64 /debug:full
OPTRPT=-Qopt-report=5

default : app


worker.o : worker.cc
	${CXX} -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

app : main.cc worker.o
	${CXX} ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" worker.obj

queue: app

	TIMEIT -m 0x1 app.exe

clean :
	rm app.exe worker.obj *.optrpt