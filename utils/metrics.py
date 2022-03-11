import timeit
import numpy as np


def basic_performance(model, timing_number=10, timing_repeat=10):
    # INPUT
    # model : GraphModule (runtime graph module that can be used to execute the graph)
    # timing_number :
    # timing_repeat :
    #
    # OUTPUT
    # perf : a dictionary with values "mean", "median" and "std"
    #
    perf = (
            np.array(timeit.Timer(lambda: model.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
    )
    perf = {
        "mean": np.mean(perf),
        "median": np.median(perf),
        "std": np.std(perf),
    }
    return perf
