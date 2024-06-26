
import gzip
import re
import os

# permissions (allow or deny) -- index 1 (2nd item in entry)
# title -- index 11 (12th item in entry)
# pub date -- index 16 (17th item in entry)
# language -- index 18 (19th item in entry)

rx_all = re.compile("^([^\t]*\t){25}[^\t\n]*\n$")
rx_allow_eng = re.compile("^[^\t]*\tallow\t([^\t]*\t){16}eng*\t([^\t]*\t){6}[^\t\n]*\n$")
date_str = "[^\t\d]*([^\t\d]|^)(184[8-9]|18[5-9]\d|19\d\d)([^\t\d]|$)[^\t]*" #matches valid year in a larger string

date_simple = "(184[8-9]|18[5-9]\d|19\d\d)"
rx = re.compile("^([^\t]*\t){16}"+date_simple)

#integrates matching of valid pub dates with matching of valid permissions and language
rx_combine = re.compile("^[^\t]*\tallow\t([^\t]*\t){14}"+date_simple+"\t[^\t]*\t"+"eng*\t([^\t]*\t){6}[^\t\n]*\n$")

count = 0
with gzip.open(os.path.expanduser("~/corpora/rx_filtered_allow_eng.tsv.gz"), "rt") as f:
	with gzip.open(os.path.expanduser("~/corpora/rx_filtered_date.tsv.gz"), "wt") as filtered:
		for doc in f:
			if re.search(rx, doc):
				count += 1
				filtered.write(doc)
			#print(doc)
			#if count == 20:
			#       break
print(count)


