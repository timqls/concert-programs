
import gzip
import re
import os

rx_all = re.compile("^([^\t]*\t){25}[^\t\n]*\n$")
rx = re.compile("^[^\t]*\tallow\t([^\t]*\t){16}eng*\t([^\t]*\t){6}[^\t\n]*\n$")
with gzip.open(os.path.expanduser("~/corpora/hathi_trust/hathi_index.tsv.gz"), "rt") as f:
	with gzip.open(os.path.expanduser("~/corpora/rx_filtered_allow_eng.tsv.gz"), "wt") as filtered:
		count = 0
		for doc in f:
			if re.search(rx, doc):
				#if True:
				count += 1
				filtered.write(doc)
				#print(doc)
				#if (count == 100):
				#       break
		print(count)



