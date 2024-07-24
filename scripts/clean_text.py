import gzip
import json
import re
import os.path
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--input", dest="input", help="Input file") # ~/corpora/concert_programs.json.gz
	parser.add_argument("--output", dest="output", help="Output file") # ~/corpora/concert_programs_cleaned.json.gz
	parser.add_argument("--min_length", dest="min_length", default=3, type=int)
	args = parser.parse_args()

	with gzip.open(args.input, "rt") as ifd, \
		gzip.open(args.output, "wt") as ofd:
		for doc in ifd:
			j = json.loads(doc)
			tokens = [re.sub(r"\W", "", w.lower()) for w in re.sub(r"\s*\-\s*", "", j["content"]).split()]
			j["content"] = " ".join([t for t in tokens if len(t) >= args.min_length])
			ofd.write(json.dumps(j) + "\n")
