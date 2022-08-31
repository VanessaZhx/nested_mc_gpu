# Main Implementation of *GPUs for Risk Management*

*Vanessa Zhu*

*Computing (Management and Finance), Imperial College London*

----

This is the main implementation of MSc Individual Project *GPUs for Risk Management* . This repository contains the GPU baseline implementation and the optimised implementations.
- For the CPU implementation, see <https://github.com/VanessaZhx/nested_mc_cpu>
- For the Moro's ICDF implementation test, see <https://github.com/VanessaZhx/MoroInverseTest>

## Install
This project uses [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [Intel MKL](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html). Please ensure they are locally installed and change the directory in `CMakeLists.txt` accordingly.

Clone the repository.
```console
$ git clone https://github.com/VanessaZhx/nested_mc_gpu.git
```
Build a folder for compiling.
```console
$ mkdir ./build
$ cd build
```

Compile the code.
```console
$ cmake ..
$ cmake --build .
```

Run the code by file `output`.
```console
$ ./output 1024 1024 10
```

## Usage
```console
$ ./output -h

===================== USAGE =====================
	-c	Use combined sobol RNG and normal transfer
	-e	Use early stop strategy for barrier option
	-s	Use same RN for inner loop
Enter up to 3 numbers for [path_ext, path_int, exp_times]
Default setup: [10, 10, 0]
```
Sample output:
```console
$ ./output 10 10 100

== SET UP ==
Experiment Times: 100
Path External: 10
Path Internal: 10
Optimisation: Combined RNG - 0
              Barrier Early Stop - 0
              Same Inner RN - 0

== DEVICE ==
Number of CUDA devices 1.
There is 1 device supporting CUDA
For device #0
Device name:                NVIDIA GeForce GTX TITAN X
Major revision number:      5
Minor revision Number:      2
Total Global Memory:        12805734400
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1215500
Number of muliprocessors:   24

== EXECUTION ==
Experiment # 0 finished.
Experiment # 10 finished.
Experiment # 20 finished.
Experiment # 30 finished.
Experiment # 40 finished.
Experiment # 50 finished.
Experiment # 60 finished.
Experiment # 70 finished.
Experiment # 80 finished.
Experiment # 90 finished.
Experiment # 99 finished.

== RESULT ==
AVG EXECUTION TIME: 1.63678 ms.
```

```console
$ ./output 10 10 100 -c -e -s

== SET UP ==
Experiment Times: 100
Path External: 10
Path Internal: 10
Optimisation: Combined RNG - 1
              Barrier Early Stop - 1
              Same Inner RN - 1

== DEVICE ==
Number of CUDA devices 1.
There is 1 device supporting CUDA
For device #0
Device name:                NVIDIA GeForce GTX TITAN X
Major revision number:      5
Minor revision Number:      2
Total Global Memory:        12805734400
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1215500
Number of muliprocessors:   24

== EXECUTION ==
Experiment # 0 finished.
Experiment # 10 finished.
Experiment # 20 finished.
Experiment # 30 finished.
Experiment # 40 finished.
Experiment # 50 finished.
Experiment # 60 finished.
Experiment # 70 finished.
Experiment # 80 finished.
Experiment # 90 finished.
Experiment # 99 finished.

== RESULT ==
AVG EXECUTION TIME: 1.71079 ms.
```

## Structure
`kernel.cu` - The main entrance to the model. Contains all the initialising parameters.

`NestedMonteCarloVaR.cu` - The main implementation of MC. 

`CUDAPricing.cu` - Kernel function implementations. 

`RNG.cu` - Random number generator. 

`Stock.h` `Bond.h` `BasketOption.h` `BarrierOption.h` - Products classes. 

`cuda_helper.cuh` - Helper functions.
