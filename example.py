# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from tvm.contrib.download import download_testdata
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

import onnx
from PIL import Image

def resnet_example():
    # Use a breakpoint in the code line below to debug your script.

    # 1) LOAD THE MODEL
    # load the model you want to optimize

    model_url = "".join(
        [
            "https://github.com/onnx/models/raw/",
            "master/vision/classification/resnet/model/",
            "resnet50-v2-7.onnx",
        ]
    )
    model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
    onnx_model = onnx.load(model_path)

    # 2) LOAD AND PREPROCESS THE IMAGE
    # load the image on which you want to run the model

    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # convert the image to a numpy array in order to use it as input

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)

    # 3) COMPILE THE MODEL WITH RELAY
    # specify the correct target (which depends on the CPU/GPU you are running)

    target = "llvm"

    # import the model to Relay, build the model into a TVM library
    # create a TVM graph

    # The input name may vary across model types. You can use a tool
    # like Netron to check input names
    input_name = "data"
    shape_dict = {input_name: img_data.shape}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    # 4) EXECUTE ON THE TVM RUNTIME
    # you just need te compiled model and the valid input to the model

    dtype = "float32"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    # 5) COLLECT UNOPTIMIZED PERFORMANCES
    # in this case we run computation in multiple batches
    # and we gather some basic statistics on the mean, median and standard deviation

    import timeit

    timing_number = 10
    timing_repeat = 10
    unoptimized = (
            np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
    )
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }
    print(f"These are the unoptimized performances: {unoptimized}")

    # 6) POSTPROCESS THE OUTPUT
    # each model will have its own particular way of providing output tensors

    from scipy.special import softmax

    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    # Open the output and read the output tensor
    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

    # this should produce the following output:
    # class='n02123045 tabby, tabby cat' with probability=0.610553
    # class='n02123159 tiger cat' with probability=0.367179
    # class='n02124075 Egyptian cat' with probability=0.019365
    # class='n02129604 tiger, Panthera tigris' with probability=0.001273
    # class='n04040759 radiator' with probability=0.000261

    # 7) TUNE THE MODEL
    # the model is optimized to run faster on a given target
    # ( it does not affect the accuracy of the model )
    # you need to provide the target specification, the path to an output file
    # and a path to the model to be tuned

    import tvm.auto_scheduler as auto_scheduler
    from tvm.autotvm.tuner import XGBTuner # this is the default tuning search algorithm (also others are available)
    from tvm import autotvm

    #set up the runner

    number = 10 # number of different config we will test
    repeat = 1 # how many measurement of each config
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0. It is useful for GPU optimization
    timeout = 10  # upper limit on how long to run training code for each config (in seconds)

    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )

    # create a simple structure for holding tuning options

    tuning_option = {
        "tuner": "xgb",
        "trials": 10, #recommended 1500 for CPU and 3000-4000 for GPU
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "resnet-50-v2-autotuning.json",
    }

    # begin by extracting the tasks from the onnx model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )
    #the output will look something like this:
    # [Task  1/24]  Current/Best:   10.71/  21.08 GFLOPS | Progress: (60/1000) | 111.77 s Done.
    # [Task  1/24]  Current/Best:    9.32/  24.18 GFLOPS | Progress: (192/1000) | 365.02 s Done.
    # [Task  2/24]  Current/Best:   22.39/ 177.59 GFLOPS | Progress: (960/1000) | 976.17 s Done.
    # [Task  3/24]  Current/Best:   32.03/ 153.34 GFLOPS | Progress: (800/1000) | 776.84 s Done.
    # [Task  4/24]  Current/Best:   11.96/ 156.49 GFLOPS | Progress: (960/1000) | 632.26 s Done.
    # [Task  5/24]  Current/Best:   23.75/ 130.78 GFLOPS | Progress: (800/1000) | 739.29 s Done.
    # [Task  6/24]  Current/Best:   38.29/ 198.31 GFLOPS | Progress: (1000/1000) | 624.51 s Done.
    # [Task  7/24]  Current/Best:    4.31/ 210.78 GFLOPS | Progress: (1000/1000) | 701.03 s Done.
    # [Task  8/24]  Current/Best:   50.25/ 185.35 GFLOPS | Progress: (972/1000) | 538.55 s Done.
    # [Task  9/24]  Current/Best:   50.19/ 194.42 GFLOPS | Progress: (1000/1000) | 487.30 s Done.
    # [Task 10/24]  Current/Best:   12.90/ 172.60 GFLOPS | Progress: (972/1000) | 607.32 s Done.
    # [Task 11/24]  Current/Best:   62.71/ 203.46 GFLOPS | Progress: (1000/1000) | 581.92 s Done.
    # [Task 12/24]  Current/Best:   36.79/ 224.71 GFLOPS | Progress: (1000/1000) | 675.13 s Done.
    # [Task 13/24]  Current/Best:    7.76/ 219.72 GFLOPS | Progress: (1000/1000) | 519.06 s Done.
    # [Task 14/24]  Current/Best:   12.26/ 202.42 GFLOPS | Progress: (1000/1000) | 514.30 s Done.
    # [Task 15/24]  Current/Best:   31.59/ 197.61 GFLOPS | Progress: (1000/1000) | 558.54 s Done.
    # [Task 16/24]  Current/Best:   31.63/ 206.08 GFLOPS | Progress: (1000/1000) | 708.36 s Done.
    # [Task 17/24]  Current/Best:   41.18/ 204.45 GFLOPS | Progress: (1000/1000) | 736.08 s Done.
    # [Task 18/24]  Current/Best:   15.85/ 222.38 GFLOPS | Progress: (980/1000) | 516.73 s Done.
    # [Task 19/24]  Current/Best:   15.78/ 203.41 GFLOPS | Progress: (1000/1000) | 587.13 s Done.
    # [Task 20/24]  Current/Best:   30.47/ 205.92 GFLOPS | Progress: (980/1000) | 471.00 s Done.
    # [Task 21/24]  Current/Best:   46.91/ 227.99 GFLOPS | Progress: (308/1000) | 219.18 s Done.
    # [Task 22/24]  Current/Best:   13.33/ 207.66 GFLOPS | Progress: (1000/1000) | 761.74 s Done.
    # [Task 23/24]  Current/Best:   53.29/ 192.98 GFLOPS | Progress: (1000/1000) | 799.90 s Done.
    # [Task 24/24]  Current/Best:   25.03/ 146.14 GFLOPS | Progress: (1000/1000) | 1112.55 s Done.

    # 8) COMPILING AN OPTIMIZED MODEL WITH TUNING DATA
    # as an output of the tuning process we obtained the tuning records
    # the compiler will use these results to generate high performance code for the model on the specific target

    # re-compile the model using optimized operators

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    # verify that the optimized model runs and produces the same results

    dtype = "float32"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

    # predictions should be the following:
    # class='n02123045 tabby, tabby cat' with probability=0.610550
    # class='n02123159 tiger cat' with probability=0.367181
    # class='n02124075 Egyptian cat' with probability=0.019365
    # class='n02129604 tiger, Panthera tigris' with probability=0.001273
    # class='n04040759 radiator' with probability=0.000261

    # 9) COMPARE TUNED AND UNTUNED MODELS

    import timeit

    timing_number = 10
    timing_repeat = 10
    optimized = (
            np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
    )
    optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}

    print(f"Performances of the optimized model: {optimized}")
    print(f"Performances of the unoptimized model: {unoptimized}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    resnet_example()

