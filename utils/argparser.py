import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    # add different options
    parser.add_argument("--model_path", help="Path of the model to be optimized with tvmc",
                        default="models/resnet50-v2-7.onnx", type=str)
    parser.add_argument("--tuning_log",
                        help="Path to the tuning log to pass as tuning_records variable in the tuning process",
                        default="logs/resnet50.log", type=str)
    parser.add_argument("--target",
                        help="Target you want to optimize the model for. Specify also the --device accordingly.",
                        default="llvm", type=str)
    parser.add_argument("--device", help="Device on which you want to run the model", default="cpu", type=str)
    parser.add_argument("--tuning_trials",
                        help="The number of schedules to try out for the entire model. "
                             "Note that the default value is chosen as a decent average for most models, "
                             "but larger models may need more trials to reach a good result while smaller models will "
                             "converge with fewer trials.", default=1000, type=int)
    parser.add_argument("--tuner",
                        help="The type of tuner to use when tuning with autotvm. "
                             "Can be one of [ga, gridsearch, random, xgb, xgb_knob and xgb-rank",
                        choices=["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb_rank"], default="xgb", type=str)
    parser.add_argument("--timeout", help="If a kernel trial lasts longer than this duration in seconds, "
                                          "it will be considered a failure", default=10, type=int)
    parser.add_argument("--repeat", help="How many times each measurement should be repeated.", default=1, type=int)

    return parser
