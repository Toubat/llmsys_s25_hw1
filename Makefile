update:
	python -m pip install -Ue .
	python -m pip install -U .


cuda-compile:
	nvcc -g -G -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
