NVCC?=nvcc

NVCCFLAGS=-O3 --expt-relaxed-constexpr -DKITTENS_ADA_LOVELACE -DNDEBUG -shared -std=c++20 -Xcompiler -fPIC -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -MD -MT -MF -x cu -lrt -lpthread -ldl -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I../include -lineinfo
TARGET=4090.so
SRC=4090.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)