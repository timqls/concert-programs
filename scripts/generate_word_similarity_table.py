import gzip
import logging
import os.path
import numpy
import argparse
import pandas
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

logger = logging.getLogger("generate_word_similarity_table")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--model", dest="model", help="Model file")
	parser.add_argument("--embeddings", dest="embeddings", help="W2V embeddings file") # work/word_2_vec_embeddings.bin
	parser.add_argument("--output", dest="output", help="File to save table") # work/word_similarity.tex
	parser.add_argument("--top_neighbors", dest="top_neighbors", default=10, type=int, help="How many neighbors to return")
	parser.add_argument('--target_words', default=["bach", "brahms", "tchaikovsky", "violin", "conductor", "love", "america"], nargs="*", help='Words to consider')
	args = parser.parse_args()

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	if args.model:
		from detm import DETM
		import torch
		with gzip.open(args.model, "rb") as ifd:
			model = torch.load(ifd, map_location=torch.device("cpu"))
		token2id = {v : k for k, v in model.id2token.items()}
		id2token = model.id2token
		model.to("cpu")
		try:
			embs = list(model.rho.parameters())[0].detach().numpy()
		except:
			embs = model.rho.cpu().detach().numpy()
		sims = cosine_similarity(embs)
	else:
		w2v = Word2Vec.load(args.embeddings)

	neighbors = []
	for w in args.target_words:
		row = [w]
		if args.model:
			i = token2id[w]
			for j in list(reversed(numpy.argsort(sims[i]).tolist()))[1:1 + args.top_neighbors]:
				ow = id2token[j]
				op = sims[i][j]
				row.append("{}:{:.02f}".format(ow, op))
		else:
			for ow, op in w2v.wv.most_similar(w, topn=args.top_neighbors):
				row.append("{}:{:.02f}".format(ow, op))
		neighbors.append(row)


	pd = pandas.DataFrame(neighbors)
	with open(args.output, "wt") as ofd:
		ofd.write(pd.to_latex(index_names=False, index=False, header=False))
