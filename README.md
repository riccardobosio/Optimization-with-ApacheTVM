# NN Optimization with Apache-TVM

## Installation 

The easiest way to install TVM is to use the 
[TLCPack](https://tlcpack.ai/). 
With this kind of installation you can't run experiments with CUDA.
It is possible that having a very recent python version does not allow you to install the TLCPack. 
Anyway with python 3.7 it works.

Otherwise you can install the library following this guide for 
[installation from source](https://tvm.apache.org/docs/install/from_source.html#install-from-source).
It is very important to modify the _config.cmake_ file, in order to set up 
all the different functionalities that you need. You can find this file in 
_tvm/cmake_ and then follow the guide instructions to edit it in the correct way.

## TVM overview

![The TVM stack](fig/The%20TVM%20stack.png)

There is one class for each required stage of a TVM workflow.

**Load**: what happens here is that a model is loaded from a supported framework and it is converted into an equivalent 
relay representation. The result of importing a model into TVM is a TVMCModel, which contains the precompiled graph 
definition and parameters that define what the model does.
All frameworks support overwriting the input shapes with a shape_dict argument. For most frameworks this is optional, 
but for Pytorch this is necessary as TVM cannot automatically search for it.

**Tune**: in the tuning process there are the compute expressions, that basically tell us which are the operations and 
how the output is computed, and there are the schedules, that are ways in which these expressions can be rewritten.
TVM optimizes across multiple layers in the following way: it looks at the model, it divides it in multiple workloads
and then optimize them. For each workload a set of possible schedules is generated and a final best schedule is selected.
A tuning table is produced: low latency and high GFLOPS are better.

**Compile**: compiling a TVMCModel produces a TVMCPackage, which contains the generated artifacts that allow the model 
to be run on the target hardware. Here there are more informations about 
[targets](https://tvm.apache.org/docs/reference/api/python/target.html).

**Run**: running a TVMCPackage produces a TVMCResult, which contains the outputs of the model and the measured runtime.
In this step you can pass, as hyperparameter _inputs_, a dictionary that maps input names to numpy values. 
If not provided, inputs will be generated using the _fill_mode_ argument. _fill_mode_ valid options are 
[zeros, ones, random] and the default is random.

## Project structure

In _main.py_ there is the code to run the experiments. 

In _utils/argparser.py_ there is the parser used to manage what is passed from
command line. 
To have more informations about what can be passed to _main.py_, you can type
```console
python3 main.py --help
```
As default, the experiment will be run using the onnx pretrained resnet50 that can be downloaded with the following
command 
```console
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx
```
and then placed in the _models_ folder. 

The same can be done for the onnx pretrained mobilenet, using this command
```console
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
```

The results of the tuning process can be found in the _logs_ folder.

## Results

These are the results of the benchmarking of different models. All the models are pretrained _.onnx_ versions.

| Model                    | Mean (ms) | Median (ms) | Max (ms) | Min (ms) | Std (ms) |
|:-------------------------|:---------:|:-----------:|:--------:|:--------:|---------:|
| Alexnet                  |  112.45   |   106.54    |  160.65  |  91.38   |    19.04 |
| **Alexnet - OPTIMIZED**      |   57.16   |    55.73    |  64.90   |  53.25   |     3.95 |
| Inception v2             |  158.72   |   157.24    |  172.35  |  156.39  |     4.57 |
| **Inception v2 - OPTIMIZED** |  118.02   |   117.50    |  120.50  |  117.27  |     1.12 |
| Resnet 18                |  154.45   |   150.98    |  189.70  |  135.38  |    17.93 |
| **Resnet 18 - OPTIMIZED**    |   98.44   |    98.39    |  99.03   |  98.31   |     0.20 |
| Resnet 50                |  325.95   |   324.57    |  335.16  |  323.23  |     3.33 |
| **Resnet 50 - OPTIMIZED**    |  227.69   |   227.54    |  230.20  |  226.96  |     0.88 |
| Resnet 101               |  655.36   |   648.70    |  692.52  |  639.17  |    17.73 |
| **Resnet 101 - OPTIMIZED**   |  467.00   |   464.49    |  488.38  |  459.98  |     8.49 |
| Densenet                 |  243.50   |   242.98    |  246.35  |  242.63  |     1.18 |
| **Densenet - OPTIMIZED**     |  170.37   |   170.33    |  170.63  |  170.23  |     0.12 |
| Googlenet                |  181.84   |   174.87    |  250.09  |  138.03  |    33.47 |
| **Googlenet - OPTIMIZED**    |   97.06   |    96.67    |  100.69  |  96.43   |     1.21 |

This is the result of testing Resnet 50 from different frameworks (_onyx_ and _keras_):

| Model                         | Mean (ms) | Median (ms) | Max (ms) | Min (ms) | Std (ms) |
|:------------------------------|:---------:|:-----------:|:--------:|:--------:|---------:|
| Resnet 50 (.onnx)             |  325.95   |   324.57    |  335.16  |  323.23  |     3.33 |
| Resnet 50 (.onnx) - OPTIMIZED |  227.69   |   227.54    |  230.20  |  226.96  |     0.88 |
| Resnet 50 (keras)             |  766.67   |   759.07    |  905.66  |  617.21  |    82.22 |
| Resnet 50 (keras) - OPTIMIZED |  183.82   |   177.07    |  215.32  |  174.26  |    13.20 |

What emerges from this test is that the keras version at the beginning is slower once loaded in TVM. However it is then 
able to reach a better optimized version since it becomes faster after the optimization. 
It can be interesting to simulate other experiments and see if this trend is followed also by the other models.
