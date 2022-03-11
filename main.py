from utils import argparser
from tvm.driver import tvmc


def optimize_tvmc(opts):
    print("Starting optimization with tvmc...")

    # step 1: load the model
    # This step converts a machine learning model from a supported framework into TVM’s high level graph representation
    # language called Relay
    print("Loading the model...")
    model = tvmc.load(opts.model_path)

    # Calculate performances of the unoptimized model
    print("Calculating performances of the unoptimized model...")
    unoptimized_package = tvmc.compile(model, target=opts.target)
    unoptimized_result = tvmc.run(unoptimized_package, device=opts.device)

    # step 1.5: optional tune
    # Run speed can further be improved by tuning
    # This optional step uses machine learning to look at each operation within a model (a function)
    # and tries to find a faster way to run it
    print("Starting tuning...")
    tvmc.tune(model, target=opts.target, tuning_records=opts.tuning_log, enable_autoscheduler=True,
              trials=opts.tuning_trials, tuner=opts.tuner, repeat=opts.repeat)

    # step 2: compile
    # This compilation process translates the model from Relay into a lower-level language
    # that the target machine can understand
    print("Calculating performances of the optimized model...")
    package = tvmc.compile(model, target=opts.target, tuning_records=opts.tuning_log)

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
