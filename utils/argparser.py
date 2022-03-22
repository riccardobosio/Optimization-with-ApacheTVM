import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    # add different options

    parser.add_argument("--model_path", help="Path of the model to be optimized with tvmc",
                        default="models/mobilenetv2-7.onnx", type=str)

    # tuning parameters

    parser.add_argument("--tuning", help="Decide if to do tuning or not.", choices=[True, False], default=True)
    parser.add_argument("--tuning_log",
                        help="Path to the tuning log to pass as tuning_records variable in the tuning process.",
                        default="logs/mobilenet.log", type=str)
    parser.add_argument("--target",
                        help="Target you want to optimize the model for. Specify also the --device accordingly.",
                        default="llvm", type=str)
    parser.add_argument("--device", help="Device on which you want to run the model.", default="cpu", type=str)
    parser.add_argument("--tuning_trials",
                        help="The number of schedules to try out for the entire model. "
                             "Note that the default value is chosen as a decent average for most models, "
                             "but larger models may need more trials to reach a good result while smaller models will "
                             "converge with fewer trials.", default=10000, type=int)
    parser.add_argument("--tuner",
                        help="The type of tuner to use when tuning with autotvm. "
                             "Can be one of [ga, gridsearch, random, xgb, xgb_knob and xgb-rank].",
                        choices=["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb_rank"], default="xgb", type=str)
    parser.add_argument("--timeout", help="If a kernel trial lasts longer than this duration in seconds, "
                                          "it will be considered a failure.", default=10, type=int)
    parser.add_argument("--repeat", help="How many times each measurement should be repeated.", default=1, type=int)

    # compilation parameters

    parser.add_argument("--compilation", help="Decide if you want to compile a model producing a TAR package or not.",
                        choices=[True, False], default=False)
    parser.add_argument("--compilation_output", help="The path where you want to save the compiled model "
                                                     "(the TAR package).", type=str,
                        default="models/resnet50-v2-7-tvm.tar")

    # inference parameters

    parser.add_argument("--inference", help="Decide if you want to make predictions on a certain input. If True,"
                                            "specify the inference_input and inference_output.",
                        choices=[True, False], default=False)
    parser.add_argument("--inference_model", help="Path to the model you want to use for inference. You should pass the"
                                                  "output of a compiled model: the output we get from the compilation "
                                                  "process is a TAR package of the model compiled to a dynamic library "
                                                  "for our target platform.", type=str,
                        default="models/resnet50-v2-7-tvm.tar")
    parser.add_argument("--inference_input", help="Pass the path of what you want to run inference on.", type=str,
                        default="data/imagenet_cat.npz")
    parser.add_argument("--inference_output", help="Pass the output path of the inference predictions.", type=str,
                        default="outputs/predictions.npz")

    return parser
