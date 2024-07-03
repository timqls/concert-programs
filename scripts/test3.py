import gzip
import re
import os


rx_str = "^([^\t]*\t){11}[^\t]*[^\t\w]festival[^\t\w][^\t]*\t"+\
        "(([^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\t)"+\
        "|(([^\t]*\t){13}[^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\n))"
rx = re.compile(rx_str, re.IGNORECASE)

count = 0
total = 0
with gzip.open(os.path.expanduser("~/projects/concert-programs/data/rx_filtered_keyword.tsv.gz"), "rt") as f:
	for doc in f:
		doc_info = doc.split("\t")
		#print(doc_info)
		total += 1
		if re.fullmatch("[\W]*", doc_info[16]):
			count += 1
			print(doc)

print(count)
print(total)
