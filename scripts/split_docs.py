import os
import json
import gzip
import argparse
import math
import random

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
	#parser.add_argument("--output_directory", dest="output_directory", help="Directory for output files") # concert_programs_split
	args = parser.parse_args()

	unique_times = set()
	total_subdocs = 0
	token_subdoc_count = {}
	data = {}

	with gzip.open(os.path.expanduser("~/corpora/" + args.input), "rt") as ifd:
		all_subdocs = []
		for doc in ifd:
			j = json.loads(doc)

			time = int(j["year"])
			unique_times.add(time)

			title = j["title"]
			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher)
			author = j.get("author", "")
			publisher = j.get("publisher", "")

			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher + ", " + "author: " + author + ", " + j["htid"])

			full_text_words = j["content"].split()

			for subdoc_num, subdoc in enumerate(split_doc(full_text_words, args.max_subdoc_len)):
				total_subdocs += 1
				utokens = set()

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
