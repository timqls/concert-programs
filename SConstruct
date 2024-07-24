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
	("HATHITRUST_ROOT", "", os.path.expanduser("~/corpora/hathi_trust")), \
	("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_index.tsv.gz"), \
	("FILTERED_INDEX", "", filtered_index_list[0]), \
	("FULL_CONTENT", "", os.path.expanduser("~/corpora/concert_programs.json.gz")), \
	("DATA_IS_LOADED", "", True), \
	("NUMS_OF_TOPICS", "", [5, 10]), \
	("WINDOW_SIZES", "", [50, 75]), \
	("MIN_WORD_OCCURRENCE", "", 60), \
	("MAX_WORD_PROP", "", 0.7), \
	("MAX_SUBDOC_LENGTHS", "", [50, 200, 400]), \
	("BATCH_SIZE", "", 1000), \
	("RANDOM_SEED", "", 1) \
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment( \
	variables=vars,\
	ENV = os.environ,
	tools = []
	BUILDERS={ \
		"FilterIndex" : Builder( \
			action="python scripts/filter_hathi_index.py --hathitrust_index ${SOURCES[0]} --output ${TARGETS[0]}"), \
		"PopulateFromIndex" : Builder( \
			action="python scripts/populate_from_index.py --hathitrust_root ${HATHITRUST_ROOT} --input ${SOURCES[0]} --output ${TARGETS[0]}"), \
		"CleanText" : Builder( \
			action="python scripts/clean_text.py --input ${SOURCES[0]} --output ${TARGETS[0]}"), \
		"TrainEmbeddings" : Builder( \
			action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"), \
		"GenerateWordSimilarityTable" : Builder( \
			action="python scripts/generate_word_similarity_table.py --embeddings ${SOURCES[0]} --output ${TARGETS[0]}"), \
		"TrainDETM" : Builder( \
			action="python scripts/train_detm.py --input ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --max_subdoc_len ${MAX_SUBDOC_LEN} --min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROP} --window_size ${WINDOW_SIZE} --random_seed ${RANDOM_SEED} --batch_size ${BATCH_SIZE} --num_topics ${NUM_TOPICS}"), \
		"ApplyDETM" : Builder( \
			action="python scripts/apply_detm.py --input ${SOURCES[0]} --model ${SOURCES[1]} --output ${TARGETS[0]} --max_subdoc_length ${MAX_SUBDOC_LEN} --batch_size ${BATCH_SIZE}"), \
		"CreateMatrices" : Builder( \
			action="python scripts/create_matrices.py --topic_annotations ${SOURCES[0]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE}"), \
		"CreateFigures" : Builder( \
			action = "python scripts/create_figures.py --input ${SOURCES[0]} --temporal_image ${TARGETS[0]} --latex ${TARGETS[1]}") \
		} \
	)

if not env["DATA_IS_LOADED"]:
	filtered = env.FilterIndex("data/hathi_index_filtered.tsv.gz", env["HATHITRUST_INDEX"])

	full_content_raw = env.PopulateFromIndex(env["FULL_CONTENT"], filtered)

	full_content_clean = env.CleanText(os.path.expanduser("~/corpora/concert_programs_cleaned.json.gz"), full_content_raw)

	embeddings = env.TrainEmbeddings("work/word_2_vec_embeddings.bin", full_content_clean)

	env.GenerateWordSimilarityTable("work/word_similarity.tex", embeddings)
else:
	full_content_clean = os.path.expanduser("~/corpora/concert_programs_cleaned.json.gz")

	embeddings = "work/word_2_vec_embeddings.bin"

for num_topics in env["NUMS_OF_TOPICS"]:
	for num_windows in env["WINDOW_SIZES"]:
		for max_subdoc_len in env["MAX_SUBDOC_LENGTHS"]:
			model = env.TrainDETM( \
				"work/detm_model_${NUM_TOPICS}_${MAX_SUBDOC_LEN}_${WINDOW_SIZE}.bin", \
				[full_content_clean, embeddings], \
				NUM_TOPICS = num_topics, \
				WINDOW_SIZE = num_windows, \
				MAX_SUBDOC_LEN = max_subdoc_len)
			labeled = env.ApplyDETM( \
				"work/results_${NUM_TOPICS}_${MAX_SUBDOC_LEN}_${WINDOW_SIZE}.json.gz", \
				[full_content_clean, model], \
				NUM_TOPICS = num_topics, \
				WINDOW_SIZE = num_windows, \
				MAX_SUBDOC_LEN = max_subdoc_len)
			matrices = env.CreateMatrices( \
				"work/matrices_${NUM_TOPICS}_${MAX_SUBDOC_LEN}_${WINDOW_SIZE}.pkl.gz", \
				labeled, \
				NUM_TOPICS = num_topics, \
				WINDOW_SIZE = num_windows, \
				MAX_SUBDOC_LEN = max_subdoc_len)

			figs = env.CreateFigures( \
				["work/temporal_image_${NUM_TOPICS}_${MAX_SUBDOC_LEN}_${WINDOW_SIZE}.png", \
				"work/tables_${NUM_TOPICS}_${MAX_SUBDOC_LEN}_${WINDOW_SIZE}.tex"], \
				matrices, \
				NUM_TOPICS = num_topics, \
				WINDOW_SIZE = num_windows, \
				MAX_SUBDOC_LEN = max_subdoc_len)


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
