import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
from steamroller import Environment


# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file.
#
# Note how, since we expect some of our build rules may want to use GPUs and/or run on
# a grid, we include a few variables along those lines that can then be overridden in
# the "custom.py" file and then used when a build rule (like "env.TrainModel") is invoked.
# Adding some indirection like this allows us finer-grained control using "custom.py",
# i.e. without having to directly edit this file.

filtered_index_list = ["data/hathi_index_filtered.tsv.gz", "data/hathi_index_filtered_more.tsv.gz"]

vars = Variables("custom.py")
vars.AddVariables(\
	("HATHITRUST_ROOT", "", "hathi_trust"), \
	("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_index.tsv.gz"), \
	("FILTERED_INDEX", "", filtered_index_list[0]), \
	("FULL_CONTENT", "", "concert_programs.json.gz"), \
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment( \
	variables=vars,\
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
	BUILDERS={ \
		"FilterIndex" : Builder(action="python scripts/filter_hathi_index.py --hathitrust_index ${HATHITRUST_INDEX} --output ${FILTERED_INDEX}"), \
		"PopulateFromIndex" : Builder( \
		action="python scripts/populate_from_index.py --hathitrust_root ${HATHITRUST_ROOT} --input ${FILTERED_INDEX} --output ${FULL_CONTENT}") \
		} \
		
	)

# At this point we have defined all the builders and variables, so it's
# time to specify the actual experimental process, which will involve
# running all combinations of datasets, folds, model types, and parameter values,
# collecting the build artifacts from applying the models to test data in a list.
#
# The basic pattern for invoking a build rule is:
#
#   "env.Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"
#
# Note how variables can be specified in each invocation, and their values used to fill
# in the build commands *and* determine output filenames, potentially overriding the global
# variables at the top of this file.  It's a very flexible system, and there are ways to
# make it less verbose, but in this case explicit is better than implicit.
#
# Note also how the outputs ("targets") from earlier invocation are used as the inputs
# ("sources") to later ones, and how some outputs are also gathered into the "results"
# variable, so they can be summarized together after each experiment runs.


env.FilterIndex([], [])
env.PopulateFromIndex([], [])

'''
results = []
for dataset_name, dataset_file in env["DATASETS"].items():
    data = env.PreprocessData("work/${DATASET_NAME}/data.txt", dataset_file, DATASET_NAME=dataset_name)
    for fold in range(1, env["FOLDS"] + 1):
        train, dev, test = env.ShuffleData(
            [
                "work/${DATASET_NAME}/${FOLD}/train.txt",
                "work/${DATASET_NAME}/${FOLD}/dev.txt",
                "work/${DATASET_NAME}/${FOLD}/test.txt",
            ],
            data,
            FOLD=fold,
            DATASET_NAME=dataset_name,
            STEAMROLLER_QUEUE=env["CPU_QUEUE"],
            STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
        )
        for model_type in env["MODEL_TYPES"]:
            for parameter_value in env["PARAMETER_VALUES"]:
                #
                # Note how the STEAMROLLER_* variables are specified differently here.
                #
                model = env.TrainModel(
                    "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/model.bin",
                    [train, dev],
                    FOLD=fold,
                    DATASET_NAME=dataset_name,
                    MODEL_TYPE=model_type,
                    PARAMETER_VALUE=parameter_value,
                    STEAMROLLER_QUEUE=env["GPU_QUEUE"],
                    STEAMROLLER_ACCOUNT=env["GPU_ACCOUNT"],
                    STEAMROLLER_GPU_COUNT=env["GPU_COUNT"]
                )
                results.append(
                    env.ApplyModel(
                        "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/applied.txt",
                        [model, test],
                        FOLD=fold,
                        DATASET_NAME=dataset_name,
                        MODEL_TYPE=model_type,
                        PARAMETER_VALUE=parameter_value,
                        STEAMROLLER_QUEUE=env["CPU_QUEUE"],
                        STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
                    )
                )
'''


# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
'''
report = env.GenerateReport(
    "work/report.txt",
    results,
    STEAMROLLER_QUEUE=env["CPU_QUEUE"],
    STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
)
'''
