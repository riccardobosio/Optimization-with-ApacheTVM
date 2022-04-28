import tvm.relay as relay
from tvm.runtime.vm import VirtualMachine
import numpy as np
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
import tvm

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

model_path = <PATH_TO_PB_FILE>

x = np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype('float32')

input_name = "main"
shape_dict = {input_name: x.shape}
dtype_dict = {"main": "float32"}

layout = None
dev = tvm.cpu(0)
target = tvm.target.Target("llvm", host="llvm")

#LOAD THE MODEL
with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict) #, outputs=out_names)

#COMPILE WITH VM
with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)
vm = VirtualMachine(vm_exec, device=dev)

#DEBUG
#func_params = vm._exec.get_function_params("main")
#print(func_params)
#var = bool()
#print(f"The value of bool() is {var}")
#print(f"The type of bool() is {type(var)}")

#EXECUTION
#1st method
######
def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

#out_node = ['detection_boxes', "detection_scores", "detection_classes"]

input_data = np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype('float32')
input_node = "Placeholder"
input_data = convert_to_list(input_data)
input_node = convert_to_list(input_node)

inputs = {}
inputs["keras_learning_phase"]=bool()
for e, i in zip(input_node, input_data):
    inputs[e] = i
result = vm.invoke("main", **inputs)
######

##2nd method
#######
#dtype = "float32"
#vm.set_input("main", keras_learning_phase=bool(), Placeholder=tvm.nd.array(x.astype(dtype)))
#vm.run(keras_learning_phase=bool(), Placeholder=tvm.nd.array(x.astype(dtype)))
#######

#GET OUTPUTS
#tvm_output = vm.get_output(0, tvm.nd.empty(((1, 1008)), "float32"))
#vm.benchmark(device=dev, keras_learning_phase=False, Placeholder=tvm.nd.array(x.astype(dtype)))
