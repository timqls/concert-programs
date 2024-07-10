import os
import json
import gzip
import argparse
import math
import random
import logging

logger = logging.getLogger("split_docs")

def split_doc(tokens, max_len):
	num_subdocs = math.ceil(len(tokens) / max_len)
	subdocs = []
	for i in range(num_subdocs):
		subdocs.append(tokens[i * max_len : (i + 1) * max_len])
	return subdocs


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", dest="input", help="Input file") # concert_programs.json.gz
	parser.add_argument("--max_subdoc_len", dest="max_subdoc_len", type=int, default=200,\
		help="Documents will be split into subdocuments of at most this number of tokens")
	parser.add_argument('--train_proportion', type=float, default=0.7, help='')
	parser.add_argument("--min_word_occurrence", dest="min_word_occurrence", type=int, default=0, \
		help="Words occuring less than this number of times throughout the entire dataset will be ignored")
	parser.add_argument("--max_word_proportion", dest="max_word_proportion", type=float, default=1.0, \
		help="Words occurring in more than this proportion of documents will be ignored (probably conjunctions, etc)")
	parser.add_argument("--window_size", dest="window_size", type=int, default=20, help="")
	#parser.add_argument("--output_directory", dest="output_directory", help="Directory for output files") # concert_programs_split
	args = parser.parse_args()

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	unique_times = set()
	total_subdocs = 0
	token_subdoc_count = {}
	data = {}
	token_id_mapping = {}

	with gzip.open(os.path.expanduser("~/corpora/" + args.input), "rt") as ifd:
		all_subdocs = []
		for doc in ifd:
			# unload each document (line) into a dictionary
			j = json.loads(doc)

			time = int(j["year"])
			unique_times.add(time)

			title = j["title"]
			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher)
			author = j.get("author", "")
			publisher = j.get("publisher", "")

			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher + ", " + "author: " + author + ", " + j["htid"])

			full_text_words = j["content"].split()

			# split into subdocs and iterate over them
			for subdoc_num, subdoc in enumerate(split_doc(full_text_words, args.max_subdoc_len)):
				total_subdocs += 1
				utokens = set()

				# get unique words in a subdoc and update counts
				for t in subdoc:
					utokens.add(t)
				for t in utokens:
					token_subdoc_count[t] = token_subdoc_count.get(t, 0) + 1

				local_dict = {\
						"time" : time, \
						"tokens" : subdoc, \
						"title" : title, \
						"id" : j["htid"] + "_" + str(subdoc_num), \
						"author" : author, \
						"publisher" : publisher, \
						"subdoc_number" : subdoc_num \
					}
				all_subdocs.append(local_dict)

		random.shuffle(all_subdocs)
		num_training = math.ceil(args.train_proportion * len(all_subdocs))

		# training/validation split
		data["train"] = all_subdocs[:num_training]
		data["val"] = all_subdocs[num_training:]

		with gzip.open(os.path.expanduser("~/corpora/concert_programs_split_train.json.gz"), "wt") as ofd_train:
			for line in data["train"]:
				ofd_train.write(json.dumps(line) + "\n")
		with gzip.open(os.path.expanduser("~/corpora/concert_programs_split_val.json.gz"), "wt") as ofd_val:
			for line in data["val"]:
				ofd_val.write(json.dumps(line) + "\n")

	vocab_kept = set()
	for t, count in token_subdoc_count.items():
		if count >= args.min_word_occurrence and (count / total_subdocs) <= args.max_word_proportion:
			vocab_kept.add(t)

	logger.info("keeping %d words from a vocabulary of %d", len(vocab_kept), len(token_subdoc_count))

	sorted_times = list(sorted(unique_times))

	min_time = sorted_times[0] - 1
	max_time = sorted_times[-1] + 1
	span = max_time - min_time

	curr_min = min_time
	curr_max = min_time

	# dictionary of window : (number of unique times in window)
	window_counts = {}

	# dictionary mapping each time to a window
	time_window_mapping = {}

	j = 0
	for i in range(math.ceil(span / args.window_size)):
		curr_max += args.window_size
		print((curr_min, curr_max))
		while j < len(sorted_times) and sorted_times[j] < curr_max:
			time_window_mapping[sorted_times[j]] = i
			j += 1
			window_counts[i] = window_counts.get(i, 0) + 1
		curr_min = curr_max

	logger.info("Found %d sub-docs, min time = %d, max time = %d, window count = %d", \
		sum([len(v) for v in data.values()]), min_time, max_time, len(window_counts))

	# dict of lists (1 for train, 1 for val) of subdocs including counts for each token
	subdoc_counts = {}
	# dict of dicts (1 dict for train, 1 for val) of window : (num subdocs in window)
	window_counts = {}

	for name, vs in data.items():
		subdoc_counts[name] = []
		window_counts[name] = {}
		for subdoc in data[name]:
			window = time_window_mapping[subdoc["time"]]
			# dict of counts for each token in subdoc
			subdoc["counts"] = {}
			subdoc["window"] = window
			for t in subdoc["tokens"]:
				if t in vocab_kept:
					tid = token_id_mapping.setdefault(t, len(token_id_mapping))
					# update counts for given token
					subdoc["counts"][tid] = subdoc["counts"].get(tid, 0) + 1
			if len(subdoc["counts"]) > 0:
				subdoc_counts[name].append(subdoc)
				window_counts[name][window] = window_counts[name].get(window, 0) + 1



	'''
	count = 0
	with gzip.open(os.path.expanduser("~/corpora/" + args.input), "rt") as ifd:
		for doc in ifd:
			count += 1
			if count == 5:
				break
			j = json.loads(doc)
			htid = j["htid"]
			content = j["content"]
			text_splitter = SpacyTextSplitter(chunk_size = 1500)
			splits = text_splitter.split_text(content)
			subdir = os.path.join(output_dir, htid)
			if not os.path.exists(subdir):
				os.mkdir(subdir)

			for i in range(len(splits)):
				with open(os.path.join(subdir, htid + "_" + str(i+1) + ".txt"), "w") as ofd:
					ofd.write(splits[i])
	'''
