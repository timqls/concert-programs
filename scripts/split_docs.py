import os
import json
import gzip
import argparse
from langchain_text_splitters import SpacyTextSplitter



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", dest="input", help="Input file") # concert_programs.json.gz
	parser.add_argument("--output_directory", dest="output_directory", help="Directory for output files") # concert_programs_split
	args = parser.parse_args()

	output_dir = os.path.expanduser("~/corpora/" + args.output_directory)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

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
