import gzip
import re
import os
#import logging
import argparse

# permissions (allow or deny) -- index 1 (2nd item in entry)
# title -- index 11 (12th item in entry)
# author -- index 12 (13th item in entry)
# pub date -- index 16 (17th item in entry)
# language -- index 18 (19th item in entry)
# publisher (?) -- index 25 (26th item in entry)

#logger = logging.getLogger("filter_hathitrust")

'''
rx_all = re.compile("^([^\t]*\t){25}[^\t\n]*\n$")
date_str = "[^\t\d]*([^\t\d]|^)(184[8-9]|18[5-9]\d|19\d\d)([^\t\d]|$)[^\t]*" #matches valid year in a larger string

date_simple = "(184[8-9]|18[5-9]\d|19\d\d)"
rx_allow_eng_date = re.compile("^[^\t]*\tallow\t([^\t]*\t){14}"+date_simple+"\t[^\t]*\teng\t([^\t]*\t){6}[^\t\n]*\n$")
'''

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--hathitrust_index", dest="hathitrust_index", help="HathiTrust index file") # hathi_trust/hathi_index.tsv.gz
	parser.add_argument("--output", dest="output", help="Output file") # data/hathi_index_filtered.tsv.gz
	args = parser.parse_args()

	rx_date = re.compile("(184[8-9]|18[5-9]\d|19\d\d)")

	# pattern for something explicitly about music performance in the document title
	music_keywords = "(music(al)?|opera(tic)?|orchestra(l)?|symphon(y|ic|ies)|philharmon(ic|ia)|chor(us|al)|choir(s)|ensemble(s)?|quartet|oratorio)"
	#authpub_keywords = "(music(al)? festival|opera(tic)?|orchestra(l)?|symphon(y|ic|ies)|philharmon(ic|ia)|chor(us|al)|choir(s)?|ensemble(s)?|quartet|oratorio)"

	# patterns to match for in title
	title_patterns = [\
		"([^\t]*[^\t\w](concerts?|recitals?|symphon(y|ic|ies)|philharmon(ic|ia))[^\t\w][^\t]*[^\t\w]program(s|me|mes)?[^\t\w][^\t]*)", \
		"([^\t]*[^\t\w]program(s|me|mes)?[^\t\w][^\t]*[^\t\w](concerts?|recitals?|oper(a|etta)|symphon(y|ic|ies)|philharmon(ic|ia))[^\t\w][^\t]*)",\
		#"[^\t\w]*programs?[^\t\w]*",\
		"([^\t]*[^\t\w]music(al)?[^\t\w][^\t]*[^\t\w]season[^\t\w][^\t]*[^\t\w]program(s|me|mes)?[^\t\w][^\t]*)",\
		"([^\t\w]*program(s)? notes[^\t\w]*)",\
		"([^\t]*[^\t\w]" + music_keywords + "[^\t\w][^\t]*[^\t\w]program(s|me|mes)?( |-)?(notes|book(s)?|bulletin(s)?)[^\t\w][^\t]*)",\
		"([^\t]*[^\t\w]program(s|me|mes)?( |-)?(notes|book(s)?|bulletin(s)?)[^\t\w][^\t]*[^\t\w]" + music_keywords + "[^\t\w][^\t]*)"\
		]
	rx_keyword_title = "("+"|".join(title_patterns)+")"

	# pattern to match for in author/publisher
	authpub_patterns = ["[^\t]*(music(al) festival)[^\t]*", "[^\t]*orchestra(l)?[^\t]*", "[^\t]*symphon(y|ic|ies)[^\t]*", \
		"[^\t]*philharmon(ic|ia)[^\t]*", "[^\t]*[^\t\w]chor(us|al)[^\t\w][^\t]*",\
		#"[^\t]*[^\t\w]choir[^\t\w][^\t]*", \
		"[^\t]*[^\t\w]quartet[^\t\w][^\t]*", "[^\t]*[^\t\w]oratorio[^\t\w][^\t]*"]
	rx_keyword_authpub = "("+"|".join(authpub_patterns)+")"


	# multi conditional pattern -- must match patterns in both title and author/publisher
	m_keywords = "(program(s|me|mes)|concert(s)|performance(s)|opera(tic)?|orchestra(l)?|"+\
		"symphon(y|ic|ies)|philharmon(ic|ia)|chor(us|al)|choir(s)|ensemble(s)?|quartet|festival|oratorio)"
	'''
	rx_multi_patterns = [\
		"^([^\t]*\t){11}[^\t]*[^\t\w]program(s|me|mes)[^\t\w][^\t]*((\t[^\t]*school of music[^\t]*\t)|(\t([^\t]*\t){13}[^\t]*school of music[^\t]*\n))",\
		"^([^\t]*\t){11}[^\t]*[^\t\w]"+m_keywords+"[^\t\w][^\t]*\t"+\
		"(([^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\t)"+\
		"|(([^\t]*\t){13}[^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\n))"\
		]
	'''
	rx_multi_patterns_tuples = [\
		(re.compile("[^\t]*[^\t\w]program(s|me|mes)[^\t\w][^\t]*", re.IGNORECASE), re.compile("[^\t]*school of music[^\t]*", re.IGNORECASE)), \
		(re.compile("[^\t]*[^\t\w]"+m_keywords+"[^\t\w][^\t]*", re.IGNORECASE), \
		re.compile("[^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*", re.IGNORECASE))]
	#rx_keyword_multi = "(" + "|".join(rx_multi_patterns) + ")"

	# multi conditional pattern incorporating all previous patterns
	'''
	rx_multi_patterns_combined = \
		[\
		"^[^\t]*\tallow\t([^\t]*\t){9}[^\t]*[^\t\w]program(s|me|mes)[^\t\w][^\t]*"+\
		"((\t[^\t]*school of music[^\t]*\t([^\t]*\t){3}"+date_simple+"\t[^\t]*\teng\t)|"+\
		"(\t([^\t]*\t){4}"+date_simple+"\t[^\t]*\teng\t([^\t]*\t){6}[^\t]*school of music[^\t]*\n))",\
		"^[^\t]*\tallow\t([^\t]*\t){9}[^\t]*[^\t\w]"+m_keywords+"[^\t\w][^\t]*\t"+\
        	"(([^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\t([^\t]*\t){3}"+date_simple+"\t[^\t]*\teng\t)"+\
        	"|(([^\t]*\t){4}"+date_simple+"\t[^\t]*\teng\t([^\t]*\t){6}[^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\n))"\
		]
	rx_keyword_multi_combined = "(" + "|".join(rx_multi_patterns_combined) + ")"
	'''
	def check_multi_patterns(tle, auth, pub):
		for title_pat, ap_pat in rx_multi_patterns_tuples:
			if re.match(title_pat, tle) and (re.match(ap_pat, auth) or re.match(ap_pat, pub)):
				return True
		return False

	# combining all keyword patterns
	#rx_str = "^([^\t]*\t){11}"+rx_keyword_title+"\t|^([^\t]*\t){12}(("+rx_keyword_authpub+"\t)|(([^\t]*\t){13}"+rx_keyword_authpub+"\n))|"+rx_keyword_multi
	#rx = re.compile(rx_str, re.IGNORECASE)

	# combining over all previous patterns (allow, english, pub date 1848-1999, keywords)
	'''
	rx_combined_str = "^[^\t]*\tallow\t([^\t]*\t){9}"+rx_keyword_title+"\t([^\t]*\t){4}"+date_simple+"\t[^\t]*\teng\t" + "|" + \
		"^[^\t]*\tallow\t([^\t]*\t){10}(("+rx_keyword_authpub+"\t([^\t]*\t){3}"+date_simple+"\t[^\t]*\teng\t)|"+\
		"(([^\t]*\t){4}"+date_simple+"\t[^\t]*\teng\t([^\t]*\t){6}"+rx_keyword_authpub+"\n))" + "|" + rx_keyword_multi_combined
	#print(repr(rx_combined_str))
	rx_combined = re.compile(rx_combined_str, re.IGNORECASE)
	'''

	# processing index
	rx_title_compiled = re.compile(rx_keyword_title, re.IGNORECASE)
	rx_authpub_compiled = re.compile(rx_keyword_authpub, re.IGNORECASE)

	count = 0
	total = 0
	with gzip.open(os.path.expanduser("~/corpora/" + args.hathitrust_index), "rt") as f:
		with gzip.open(args.output, "wt") as filtered:
			for doc in f:
				total += 1
				items = doc.split("\t")
				if items[1] == "allow" and items[18] == "eng" and re.fullmatch(rx_date, items[16]):
					title = items[11]
					author = items[12]
					publisher = items[25]
					#print(title)
					#print(author)
					#print(publisher)
					if re.match(rx_title_compiled, title) or re.match(rx_authpub_compiled, author) or \
						re.match(rx_authpub_compiled, publisher) or check_multi_patterns(title, author, publisher):
						count += 1
						filtered.write(doc)
						#print(doc)
				#if count == 50:
				#	break
	print("Documents selected : " + str(count) + " out of " + str(total))

