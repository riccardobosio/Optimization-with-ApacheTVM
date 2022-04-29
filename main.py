from utils import argparser
from tvm.driver import tvmc
import os
from utils.postprocess import resnet50_get_probabilities


def optimize_tvmc(opts):
    print("Starting optimization with tvmc...")

    # step 1: load the model
    # This step converts a machine learning model from a supported framework into TVM’s high level graph representation
    # language called Relay
    print("Loading the model...")
    model = tvmc.load(opts.model_path)
    model.summary()

    # Calculate performances of the unoptimized model
    print("Calculating performances of the unoptimized model...")
    unoptimized_package = tvmc.compile(model, target=opts.target)

    if opts.compilation:
        print("Compiling the model and producing TAR package...")
        os.system("tvmc compile --target {0} --output {1} {2}".format(opts.target, opts.compilation_output, opts.model_path))

    if opts.inference:
        print("Running inference...")
        os.system("tvmc run --inputs {0} --output {1} {2}"
                  .format(opts.inference_input, opts.inference_output, opts.inference_model))
        print("These are the output probabilities of the UNOPTIMIZED MODEL:")
        resnet50_get_probabilities(opts.inference_output)

    unoptimized_result = tvmc.run(unoptimized_package, device=opts.device)

    # step 1.5: optional tune
    # Run speed can further be improved by tuning
    # This optional step uses machine learning to look at each operation within a model (a function)
    # and tries to find a faster way to run it
    if opts.tuning:
        print("Starting tuning...")
        tvmc.tune(model, target=opts.target, tuning_records=opts.tuning_log, enable_autoscheduler=True,
                  trials=opts.tuning_trials, tuner=opts.tuner, repeat=opts.repeat)

    # step 2: compile
    # This compilation process translates the model from Relay into a lower-level language
    # that the target machine can understand
    print("Calculating performances of the optimized model...")
    package = tvmc.compile(model, target=opts.target, tuning_records=opts.tuning_log)

    if opts.compilation:
        out_path = opts.compilation_output.split(".")[0] + "_tuned.tar"
        print("Compiling the tuned model...")
        os.system("tvmc compile --target {0} --tuning-records {1}  --output {2} {3}"
                  .format(opts.target, opts.tuning_log, out_path, opts.model_path))

    if opts.inference:
        inf_model_path = opts.inference_model.split(".")[0] + "_tuned.tar"
        inf_out_path = opts.inference_output.split(".")[0] + "_tuned.npz"
        print("Running inference with the tuned model...")
        os.system("tvmc run --inputs {0} --output {1} {2}"
                  .format(opts.inference_input, inf_out_path, inf_model_path))
        print("These are the output probabilities of the OPTIMIZED MODEL:")
        resnet50_get_probabilities(inf_out_path)

    # step 3: run
    # The compiled package can now be run on the hardware target
    result = tvmc.run(package, device=opts.device)

    print("These are the performances of the UNOPTIMIZED model:")
    print(unoptimized_result)
    print("These are the performances of the OPTIMIZED model")
    print(result)


if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    optimize_tvmc(opts)
