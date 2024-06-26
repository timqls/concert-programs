import gzip
import re
import os

# permissions (allow or deny) -- index 1 (2nd item in entry)
# title -- index 11 (12th item in entry)
# author -- index 12 (13th item in entry)
# pub date -- index 16 (17th item in entry)
# language -- index 18 (19th item in entry)
# publisher (?) -- index 25 (26th item in entry)

rx_all = re.compile("^([^\t]*\t){25}[^\t\n]*\n$")
date_str = "[^\t\d]*([^\t\d]|^)(184[8-9]|18[5-9]\d|19\d\d)([^\t\d]|$)[^\t]*" #matches valid year in a larger string

date_simple = "(184[8-9]|18[5-9]\d|19\d\d)"
rx_allow_eng_date = re.compile("^[^\t]*\tallow\t([^\t]*\t){14}"+date_simple+"\t[^\t]*\teng*\t([^\t]*\t){6}[^\t\n]*\n$")

# pattern for something explicitly about music performance
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
	"symphon(y|ic|ies)|philharmon(ic|ia)|chor(us|al)|choir(s)|ensemble(s)?|quartet|oratorio)"
rx_multi_patterns = [\
	"^([^\t]*\t){11}[^\t]*[^\t\w]program(s|me|mes)[^\t\w][^\t]*((\t[^\t]*school of music[^\t]*\t)|(\t([^\t]*\t){13}[^\t]*school of music[^\t]*\n))",\
	"^([^\t]*\t){11}[^\t]*[^\t\w]"+m_keywords+"[^\t\w][^\t]*\t"+\
	"(([^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\t)"+\
	"|(([^\t]*\t){13}[^\t]*(music(al)? (societ(y|ies)|association|club))[^\t]*\n))"\
	]
rx_keyword_multi = "(" + "|".join(rx_multi_patterns) + ")"
#print(repr(rx_keyword_multi))
#re.compile(rx_keyword_multi)

#print(rx_keyword_title)
#print(rx_keyword_authpub)

# combining all keyword patterns
rx_str = "^([^\t]*\t){11}"+rx_keyword_title+"\t|([^\t]*\t){12}(("+rx_keyword_authpub+"\t)|(([^\t]*\t){13}"+rx_keyword_authpub+"\n))|"+rx_keyword_multi
print(repr(rx_str))
rx = re.compile(rx_str, re.IGNORECASE)


# combining over all previous patterns (allow, english, pub date 1848-1999, keywords)
#rx_combined =

count = 0
total = 0
with gzip.open(os.path.expanduser("~/corpora/rx_filtered_date.tsv.gz"), "rt") as f:
	with gzip.open("data/rx_filtered_keyword.tsv.gz", "wt") as filtered:
		for doc in f:
			total += 1
			if re.search(rx, doc):
				count += 1
				filtered.write(doc)
				#print(doc)
			if total % 50000 == 0:
				print(count)
print(count)
