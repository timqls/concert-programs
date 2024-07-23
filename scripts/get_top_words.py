import json
import gzip
import argparse



if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--input", dest="input", help="input file")
	args = parser.parse_args()


	topics2wordcounts = {}
	with gzip.open(args.input, "rt") as ifd:
		for doc in ifd:
			words_topics = json.loads(doc)["text"]
			for word, topic in words_topics:
				if topic in topics2wordcounts:
					topics2wordcounts[topic][word] = topics2wordcounts[topic].setdefault(word, 0) + 1
				elif topic != None:
					topics2wordcounts[topic] = {word : 1}

	with open(args.input.split(".")[0].replace("results", "topwords") + ".txt", "w") as ofd:
		for topic, word_count_dict in sorted(topics2wordcounts.items()):
			words_counts = sorted([(cnt, wrd) for wrd, cnt in word_count_dict.items()], reverse=True)
			ofd.write("topic " + str(topic) + "\n")
			ofd.write(str(words_counts[:20]))
			ofd.write("\n\n")
