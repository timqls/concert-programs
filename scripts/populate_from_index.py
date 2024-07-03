# much of this code is borrowed from the russian semantics project populate_hathitrust.py script:
# see https://github.com/comp-int-hum/russian-semantics/blob/main/scripts/populate_hathitrust.py

import logging
import gzip
import os.path
import json
import zipfile
import re
import argparse
from pairtree import PairtreeStorageFactory


# permissions (allow or deny) -- index 1 (2nd item in entry)
# title -- index 11 (12th item in entry)
# author -- index 12 (13th item in entry)
# pub date -- index 16 (17th item in entry)
# language -- index 18 (19th item in entry)
# publisher (?) -- index 25 (26th item in entry)

logger = logging.getLogger("populate_hathitrust")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--hathitrust_root", dest="hathitrust_root", help="HathiTrust root directory")
	parser.add_argument("--input", dest="input", help="Input file")
	parser.add_argument("--output", dest="output", help="Output file")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	psf = PairtreeStorageFactory()

	with gzip.open(args.input, "rt") as ifd, gzip.open(os.path.expanduser("~/corpora/"+args.output), "wt") as ofd:
		for i, doc in enumerate(ifd):
			#if i == 6:
			#	break
			doc_info = doc.split("\t")
			htid = doc_info[0]
			title = doc_info[11]
			author = doc_info[12]
			pubdate = doc_info[16]
			publisher = doc_info[25]

			val = { \
				"title" : doc_info[11], \
				"htid" : doc_info[0] \
				}
			if author:
				val["author"] = author
			if publisher:
				val["publisher"] = publisher
			if pubdate:
				val["year"] = doc_info[16]

			print(i, val["title"])

			id_toks = htid.split(".")
			subcollection = id_toks[0]
			pairtree_name = id_toks[0].replace('/', '.')
			pairtree_path = ".".join(id_toks[1:]).replace('/', '.')
			#mid = os.path.join(pairtree_name, pairtree_path)
			ident = ".".join(id_toks[1:])
			#print(os.path.join(os.path.expanduser("~/corpora/"+args.hathitrust_root), subcollection))
			try:
				store = psf.get_store(store_dir = os.path.join(os.path.expanduser("~/corpora/"+args.hathitrust_root), subcollection))
				obj = store.get_object(ident, create_if_doesnt_exist=False)
			except Exception as e:
				logger.error("Could not access HathiTrust document '%s'", htid)
				#raise e
				pass # get_object fails for a handful of entries in the input file
			full_content = []
			for subpath in obj.list_parts():
				for fname in obj.list_parts(subpath):
					#print("fname " + fname)
					#print("subpath " + subpath+"\n\n\n")
					if fname.endswith("zip"):
						with zipfile.ZipFile(obj.get_bytestream("{}/{}".format(subpath, fname), streamable=True)) as izf:
							for page in sorted(izf.namelist()):
								if page.endswith("txt"):
									txt = izf.read(page).decode("utf-8")
									#if correct_line_breaks:
									#    txt = re.sub(r"\-\s*?\n\s*", "", txt)
									full_content.append(txt.replace("\n",  " "))
			pages = []
			for page in full_content:
				#print(page)
				pages.append(page)
			val["content"] = "\n".join(pages)
			ofd.write(json.dumps(val) + "\n")
