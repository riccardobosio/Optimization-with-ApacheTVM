from tvm.driver import tvmc
from tvm.contrib.download import download_testdata
from tensorflow import keras
import numpy as np
from PIL import Image
import tvm
import os
import glob
import cv2
import tqdm



def optimize_tf_models():

    ##KERAS.applications
    #keras_resnet101 = keras.applications.resnet.ResNet101(weights='imagenet')
    #keras_resnet101.save("models/keras_resnet101")

    #model = tvmc.load("models/keras_resnet101", model_format="keras")

    # PB MODEL
    model = tvmc.load(<PATH_TO_PROTOBUF>, model_format="pb")

    # IMAGE
    img_path = <PATH_TO_IMAGE>
    dtype = "float32"
    in_ = <INSERT_IMAGE_PREPROCESSING>
    input_dict = {"Placeholder": in_.astype(dtype)}

    print("Calculating performances of the unoptimized model...")
    unoptimized_package = tvmc.compile(model, target='llvm')

    unoptimized_result = tvmc.run(unoptimized_package, device='cpu', inputs=input_dict)

    unoptimized_tensor = unoptimized_result.outputs["output_0"]
    print(f"unoptimized_tensor is a {type(unoptimized_tensor)}")
    print(f"of shape {unoptimized_tensor.shape}")

    print("Starting tuning...")
    tvmc.tune(model, target='llvm', tuning_records="logs/optimLP_detector_15000.log", enable_autoscheduler=True)

    print("Calculating performances of the optimized model...")
    package = tvmc.compile(model, target='llvm', tuning_records="logs/optimLP_detector_15000.log")#,
                           #package_path="models/optimized_LP_detector/optimLP_detector_package")

    result = tvmc.run(package, device="cpu", inputs=input_dict)

    optimized_tensor = result.outputs["output_0"]

    print(f"optimized_tensor is a {type(optimized_tensor)}")
    print(f"of shape {optimized_tensor.shape}")

    print("These are the performances of the UNOPTIMIZED model:")
    print(unoptimized_result)
    print("These are the performances of the OPTIMIZED model")
    print(result)

    # save output vectors
    #np.save(file="data/LPdetector/numpy_output/unoptimized_out.npy", arr=unoptimized_tensor)
    #np.save(file="data/LPdetector/numpy_output/optimized_out.npy", arr=optimized_tensor)


if __name__ == "__main__":
    optimize_tf_models()