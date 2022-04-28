import tvm
from tvm import te
from tvm import relay
import numpy as np
import os.path
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
from PIL import Image
from tvm.contrib import graph_executor
from tvm import autotvm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
import timeit


try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

#DEFINE PARAMETERS
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
model_path = <INSERT_MODEL_PATH>
img_path = <INSERT_IMG_PATH>
dtype = "float32"

#IMAGE
image = Image.open(img_path).resize((512, 512))
x = np.array(image)
print(x.shape)

with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)


mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout) #, shape=shape_dict)
print("Tensorflow protobuf imported to relay frontend.")

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

unoptimized_m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
unoptimized_m.set_input("Placeholder", tvm.nd.array(x.astype(dtype))) #"Placeholder" is the name of the first block
# execute
unoptimized_m.run()
# get outputs
tvm_output_unoptim = unoptimized_m.get_output(0)#0, tvm.nd.empty(((1, 1008)), "float32"))
print(type(tvm_output_unoptim))
print(tvm_output_unoptim.shape)

#COLLECT PERFORMANCES
timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: unoptimized_m.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)

#TUNING
number = 10
repeat = 1
min_repeat_ms = 0,  # before it was set to 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 1500,  # number of trials = 1500 for CPU
    "early_stopping": 1500,  # before it was set to 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "logs/LP_autotvm.json",
}

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

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

#OPTIMIZED MODEL
with autotvm.apply_history_best("logs/LP_autotvm.json"):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

optimized_m = graph_executor.GraphModule(lib["default"](dev))

optimized_m.set_input("Placeholder", tvm.nd.array(x.astype(dtype))) #"Placeholder" is the name of the first block
# execute
optimized_m.run()
# get outputs
tvm_output_optim = optimized_m.get_output(0)#0, tvm.nd.empty(((1, 1008)), "float32"))
print(type(tvm_output_optim))
print(tvm_output_optim.shape)

#COLLECT PERFORMANCES
timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: optimized_m.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

print(f"Are the two output equals to each other? {tvm_output_unoptim.same_as(tvm_output_optim)}")
