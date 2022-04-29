import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import time
import timeit
from tvm.driver import tvmc


with tf.compat.v1.Session() as persisted_sess:
  print("Loading graph...")
  with gfile.FastGFile(<PATH_TO_PB_FILE>, 'rb') as f:
    graph = tf.compat.v1.Graph()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def)
  print("Done.")


  # IMPORT AND PREPROCESS THE IMAGE
  print("Preprocessing the image...")
  img_path = <PATH_TO_TEST_IMAGE>
  dtype = "float32"
  in_ = <ADD_PREPROCESSING>
  print("Done.")

  input_dict = {"import/Placeholder:0": in_.astype(dtype)}

  # DEBUG tensor names
  #all_tensors = [tensor for op in tf.compat.v1.get_default_graph().get_operations() for tensor in op.values()]
  #print(all_tensors)

  #compute predictions
  timing_number = 10
  timing_repeat = 10
  unoptimized = (
          np.array(timeit.Timer(lambda: persisted_sess.run("import/Output:0", feed_dict=input_dict)).repeat(repeat=timing_repeat, number=timing_number))
          * 1000
          / timing_number
  )
  unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
  }

# NOW LET'S TEST TVM RUNTIME
optim_LP = tvmc.TVMCPackage(package_path="models/optimized_LP_detector/optimLP_detector_package")
input_dict = {"Placeholder": in_.astype("float32")}

optimized = (
          np.array(timeit.Timer(lambda: tvmc.run(optim_LP, device="cpu", inputs=input_dict, number=1, repeat=1)).repeat(repeat=timing_repeat, number=timing_number))
          * 1000
          / timing_number
  )
optimized = {
    "mean": np.mean(optimized),
    "median": np.median(optimized),
    "std": np.std(optimized),
  }

result = tvmc.run(optim_LP, device="cpu", inputs=input_dict, number=1, repeat=1)

# SEE WHAT HAPPENS WITH A MODEL JUST IMPORTED
model = tvmc.load(<PATH_TO_PB_FILE>, model_format="pb")
unoptimized_package = tvmc.compile(model, target='llvm')

just_imported = (
          np.array(timeit.Timer(lambda: tvmc.run(unoptimized_package, device="cpu", inputs=input_dict, number=1, repeat=1)).repeat(repeat=timing_repeat, number=timing_number))
          * 1000
          / timing_number
  )
just_imported = {
    "mean": np.mean(just_imported),
    "median": np.median(just_imported),
    "std": np.std(just_imported),
  }

# PRINT RESULTS
print("Tensorflow performances:")
print(unoptimized)
print("Unoptimized TVM runtime performances:")
print(just_imported)
print("Optimized TVM runtime performances:")
print(optimized)
print("Optimized TVM runtime collected perf:")
print(result)
