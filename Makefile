NVCC=nvcc

####################################
# OpenCV default install locations #
# Check yours and replace.         #
####################################

OPENCV_LIBPATH=/usr/local/lib
OPENCV_INCLUDEPATH=/usr/local/include

CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -m64 -Wno-deprecated-gpu-targets `pkg-config --cflags --libs opencv`
GCC_OPTS=-O3 -m64 `pkg-config --cflags --libs opencv`

photops: main.o load_save.o blur_ops.o filter_ops.o mirror_ops.o square_ops.o Makefile
	$(NVCC) -o photops main.o load_save.o blur_ops.o filter_ops.o mirror_ops.o square_ops.o -L $(OPENCV_LIBPATH) -lboost_program_options $(NVCC_OPTS)

main.o: main.cpp include/load_save.h include/square_ops.h include/mirror_ops.h include/blur_ops.h include/filter_ops.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

load_save.o: load_save.cpp include/load_save.h
	g++ -c load_save.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

# Maybe there's no need to have $OPENCV_INCLUDEPATH in the following files.
blur_ops.o: blur_ops.cu include/load_save.h include/blur_ops.h
	$(NVCC) -c blur_ops.cu $(NVCC_OPTS)

mirror_ops.o: mirror_ops.cu include/load_save.h include/mirror_ops.h
	$(NVCC) -c mirror_ops.cu $(NVCC_OPTS)

square_ops.o: square_ops.cu include/load_save.h include/square_ops.h
	$(NVCC) -c square_ops.cu $(NVCC_OPTS)

filter_ops.o: filter_ops.cu include/load_save.h include/filter_ops.h
	$(NVCC) -c filter_ops.cu $(NVCC_OPTS)

clean:
	rm -f *.o photops
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f


