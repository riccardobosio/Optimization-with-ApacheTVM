from tvm.contrib.download import download_testdata
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

def resnet_example():
    # this example follows https://tvm.apache.org/docs/how_to/compile_models/from_keras.html

    # use an example input to trace the operations of the model
    # example_input = torch.rand(1, 3, 224, 224) # 224 is the least input size, depends on the dataset you use

    # 1) LOAD THE MODEL
    import keras
    import tensorflow as tf

    if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
        weights_url = "".join(
            [
                "https://github.com/fchollet/deep-learning-models/releases/",
                "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
            ]
        )
        weights_file = "resnet50_keras_old.h5"
    else:
        weights_url = "".join(
            [
                " https://storage.googleapis.com/tensorflow/keras-applications/",
                "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
            ]
        )
        weights_file = "resnet50_keras_new.h5"

    weights_path = download_testdata(weights_url, weights_file, module="keras")
    keras_resnet50 = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
    )
    keras_resnet50.load_weights(weights_path)

    # 2) LOAD AND PREPROCESS THE IMAGE

    # load the image on which you want to run the model

    from PIL import Image
    from matplotlib import pyplot as plt
    from keras.applications.resnet50 import preprocess_inpu
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    plt.imshow(img)
    plt.show()
    # input preprocess
    img_data = np.array(img)[np.newaxis, :].astype("float32")
    img_data = preprocess_input(img_data).transpose([0, 3, 1, 2])
    print("input_1", img_data.shape)

    # 3) COMPILE THE MODEL WITH RELAY
    # specify the correct target (which depends on the CPU/GPU you are running)

    # target = "llvm"
    target = "cuda"
    dev = tvm.cuda(0)

    # import the model to Relay, build the model into a TVM library
    # create a TVM graph
    # The input name may vary across model types. You can use a tool
    # like Netron to check input names

    # compile the model
    shape_dict = {"input_1": img_data.shape}
    mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

    # TODO(mbs): opt_level=3 causes nn.contrib_conv2d_winograd_weight_transform
    # to end up in the module which fails memory validation on cuda most likely
    # due to a latent bug. Note that the pass context only has an effect within
    # evaluate() and is not captured by create_executor().

    with tvm.transform.PassContext(
            opt_level=3):  # TRY TO SOLVE THIS ISSUE https://discuss.tvm.apache.org/t/the-type-inference-pass-was-unable-to-infer-a-type-for-this-expression/8600
        # model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
        intrp = relay.build_module.create_executor("graph", mod, dev, target, params)

    feats = np.random.rand(1, 3, 224, 224).astype(np.float32)
    tvm_input = tvm.nd.array(feats)

    print(intrp.mod)

    tvm_output = intrp.evaluate()(tvm_input, **params).asnumpy()

    # 4) EXECUTE ON THE TVM RUNTIME
    # you just need te compiled model and the valid input to the mode

    dtype = "float32"
    tvm_out = model(tvm.nd.array(data.astype(dtype)))
    top1_tvm = np.argmax(tvm_out.numpy()[0])

    # LOOKUP PREDICTION (keras guide)
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())
    print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
    # confirm correctness with keras output
    keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
    top1_keras = np.argmax(keras_out)
    print("Keras top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))

    # 5) COLLECT UNOPTIMIZED PERFORMANCES
    # in this case we run computation in multiple batches
    # and we gather some basic statistics on the mean, median and standard deviatio
    import timei
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
    # each model will have its own particular way of providing output tensor
    from scipy.special import softma
    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data"
    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f
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
    from tvm.autotvm.tuner import XGBTuner  # this is the default tuning search algorithm (also others are available)
    from tvm import autotvm

    # set up the runne
    number = 10  # number of different config we will test
    repeat = 1  # how many measurement of each config
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0. It is useful for GPU optimization
    timeout = 10  # upper limit on how long to run training code for each config (in seconds
    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,

        # create a simple structure for holding tuning option
        tuning_option={
            "tuner": "xgb",
            "trials": 10,  # recommended 1500 for CPU and 3000-4000 for GPU
            "early_stopping": 100,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"), runner=runner
            ),
            "tuning_records": "resnet-50-v2-autotuning.json",
        }

    # begin by extracting the tasks from the onnx model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params
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
    # the output will look something like this:
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
    # [Task 24/24]  Current/Best:   25.03/ 146.14 GFLOPS | Progress: (1000/1000) | 1112.55 s Done
    # 8) COMPILING AN OPTIMIZED MODEL WITH TUNING DATA
    # as an output of the tuning process we obtained the tuning records
    # the compiler will use these results to generate high performance code for the model on the specific targe
    # re-compile the model using optimized operators

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with
    tvm.transform.PassContext(opt_level=3, config={}):
    lib = relay.build(mod, target=target, params=params
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev)
    # verify that the optimized model runs and produces the same result
    dtype = "float32"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy(
        scores=softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

    # predictions should be the following:
    # class='n02123045 tabby, tabby cat' with probability=0.610550
    # class='n02123159 tiger cat' with probability=0.367181
    # class='n02124075 Egyptian cat' with probability=0.019365
    # class='n02129604 tiger, Panthera tigris' with probability=0.001273
    # class='n04040759 radiator' with probability=0.00026
    # 9) COMPARE TUNED AND UNTUNED MODEL
    import timei
    timing_number = 10
    timing_repeat = 10
    optimized = (
            np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
    )
    optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)
                 print(f"Performances of the optimized model: {optimized}")
                     print(f"Performances of the unoptimized model: {unoptimized}")

if __name__ == '__main__':
    resnet_example()

