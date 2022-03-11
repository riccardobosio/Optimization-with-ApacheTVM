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
relay representation. The result of importing a model into TVM is a TVMModel, which contains the precompiled graph 
definition and parameters that define what the model does.

**Tune**: in the tuning process there are the compute expressions, that basically tell us which are the operations and 
how the output is computed, and there are the schedules, that are ways in which these expressions can be rewritten.
TVM optimizes across multiple layers in the following way: it looks at the model, it divides it in multiple workloads
and then optimize them. For each workload a set of possible schedules is generated and a final best schedule is selected.
A tuning table is produced: low latency and high GFLOPS are better.

**Compile**: compiling a TVMCModel produces a TVMCPackage, which contains the generated artifacts that allow the model 
to be run on the target hardware. 

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

The results of the tuning process can be found in the _logs_ folder.





