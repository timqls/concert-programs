## experimenting with reducing dataset further

import re
import gzip
import argparse
import os

# permissions (allow or deny) -- index 1 (2nd item in entry)
# title -- index 11 (12th item in entry)
# author -- index 12 (13th item in entry)
# pub date -- index 16 (17th item in entry)
# language -- index 18 (19th item in entry)
# publisher (?) -- index 25 (26th item in entry)


def check_if_in(words, items):
	for item in items:
		for word in words:
			if word in item:
				return True
	return False

def check_condition(title, author, pub):
	if check_if_in(["orchestral association"], [author, pub]):
		return True
	elif check_if_in(["boston", "university of michigan", "crystal palace", "chicago", "toronto", "worcester", "birmingham"], [title, author, pub]):
		return True
	elif re.fullmatch(".*orchestra\s*(\.)?\s*", author) or re.fullmatch(".*orchestra\s*(\.)?\s*", pub):
		return True
	elif re.fullmatch(".*(symphony|philharmonic)\s*society(\W)*", author) or re.fullmatch(".*(symphony|philharmonic)\s*society(\W)*", pub):
		return True
	return False

years_counts = {1852 : 0, 1902 : 0, 1952 : 0}
count = 0
with gzip.open("data/hathi_index_filtered.tsv.gz", "rt") as ifd, gzip.open("data/hathi_index_filtered_more.tsv.gz", "wt") as ofd:
	for doc in ifd:
		items = doc.split("\t")
		title = items[11].lower()
		author = items[12].lower()
		pub = items[25].lower()
		year = int(items[16])
		#if year >= 1972:
		#	print(doc)
		#	count += 1
		#if check_if_in(["london"], [title, author, pub]):
		if check_condition(title, author, pub):
			count += 1
			ofd.write(doc)
			for key in sorted(years_counts.keys(), reverse = True):
				if year >= key:
					years_counts[key] += 1
					break
		else:
			if check_if_in(["toronto", "worcester", "birmingham"], [title, author, pub]):
				print(doc)
print(count)
print(years_counts)
